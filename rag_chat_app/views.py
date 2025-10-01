import json
import os
import uuid
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_http_methods
from django.contrib.auth.models import User
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from google import genai 
from google.genai.errors import APIError 

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document as DocxDocument

EMBEDDING_MODEL = "gemini-embedding-001" 
LLM_MODEL = "gemini-2.5-flash"
HISTORY_LIMIT = 5 
CHROMA_PATH = "./chroma_db" 

try:
    from .models import ChatThread, Message, Document
except ImportError:
    class Document:
        def __init__(self):
            self.id = 1
            self.filename = "placeholder.pdf"
            self.collection_name = ""
            self.user = None
            self.is_ready = False
        def delete(self): pass
        def save(self): pass
        @staticmethod
        def objects(): 
            class Manager:
                def first(self): return User.objects.first()
                def create(self, **kwargs): return Document()
                def get(self, **kwargs): return Document()
                def filter(self, **kwargs): return []
            return Manager()
    class ChatThread:
        def __init__(self): self.id = 1
        def delete(self): pass
        @staticmethod
        def objects(): 
            class Manager:
                def create(self, **kwargs): return ChatThread()
                def get(self, **kwargs): return ChatThread()
                def filter(self, **kwargs): return []
            return Manager()
    class Message:
        def __init__(self): pass
        @staticmethod
        def objects(): 
            class Manager:
                def create(self, **kwargs): pass
                def filter(self, **kwargs): return []
                def exclude(self, **kwargs): return self
                def order_by(self, **kwargs): return []
                def values(self, *args): return []
            return Manager()
    class User:
        def __init__(self): self.id = 1
        @staticmethod
        def objects(): 
            class Manager:
                def create_user(self, **kwargs): return User()
                def first(self): return User()
                def get(self, **kwargs): return User()
            return Manager()

def get_gemini_client():
    """
    Initializes and returns the Gemini client with the API key 
    from the GEMINI_API_KEY environment variable.
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        # Note: This is crucial for debugging the API key issue.
        raise ValueError("GEMINI_API_KEY environment variable not set. Cannot initialize Gemini client.")
    
    return genai.Client(api_key=api_key)

def get_chroma_client():
    """Returns the persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PATH)

def extract_text_from_file(file_obj) -> str:
    """Extracts text from various file types (PDF, DOCX, TXT)."""
    text = ""
    file_name = file_obj.name.lower()

    if file_name.endswith('.pdf'):
        reader = PdfReader(file_obj)
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    elif file_name.endswith(('.doc', '.docx')):
        doc = DocxDocument(file_obj)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    elif file_name.endswith('.txt'):
        file_obj.seek(0) 
        text = file_obj.read().decode('utf-8')
    
    return text

def index_document_in_chroma(document_obj, raw_text: str) -> bool:
    """
    Chunks text, generates Gemini embeddings explicitly, and inserts them 
    into a new Chroma collection.
    """
    collection_name = f"doc-{document_obj.id}-{uuid.uuid4().hex[:8]}"
    
    # 1. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300, 
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_text(raw_text)
    
    if not texts:
        return False

    try:
        # 2. Initialize Gemini Client and Generate Embeddings
        client = get_gemini_client() 
        
        # Call the embed_content API
        print(f"Generating embeddings for {len(texts)} chunks using {EMBEDDING_MODEL}...")
        embedding_response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        
        embeddings = [e.values for e in embedding_response.embeddings] 
        
        if len(embeddings) != len(texts):
            print("Gemini embedding failed to return vectors for all chunks.")
            return False
            
        chroma_client = get_chroma_client()
        collection = chroma_client.create_collection(name=collection_name) 
        
        collection.add(
            documents=texts,
            embeddings=embeddings, 
            ids=[f"{document_obj.id}-{i}" for i in range(len(texts))]
        )
        
        document_obj.collection_name = collection_name
        document_obj.is_ready = True
        document_obj.save()
        return True
    except ValueError as e:
        print(f"Gemini Client Error: {e}")
        return False
    except APIError as e:
        print(f"Gemini Embedding API Error: {e}")
        return False
    except Exception as e:
        print(f"Chroma indexing or general error: {e}")
        return False

def index(request):
    """Renders the single-page chat interface."""
    user = User.objects.first() 
    if not user:
        user = User.objects.create_user(username='demo_user', password='password123')
    threads = ChatThread.objects.filter(user=user).order_by('-created_at')
    return render(request, 'rag_chat_app/index.html', {'user_id': user.id, 'threads': threads})

@csrf_exempt
@require_POST
def create_thread(request):
    """Creates a new chat thread."""
    try:
        user = User.objects.get(id=json.loads(request.body).get('user_id'))
        thread = ChatThread.objects.create(user=user, title="New Chat")
        
        Message.objects.create(
            thread=thread, 
            sender='assistant', 
            content='Welcome! Upload a document to ask grounded questions, or start a general chat.'
        )
        return JsonResponse({'thread_id': thread.id, 'title': thread.title})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def get_messages(request, thread_id):
    """Retrieves all messages for a given thread."""
    try:
        thread = ChatThread.objects.get(id=thread_id)
        messages = Message.objects.filter(thread=thread).values('sender', 'content', 'is_rag')
        return JsonResponse({'messages': list(messages), 'title': thread.title})
    except ChatThread.DoesNotExist:
        return JsonResponse({'error': 'Thread not found'}, status=404)

class DocumentListCreateView(APIView):
    """Handles document upload and listing."""
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, format=None):
        """List all ready documents for the current user."""
        user = User.objects.first()
        documents = Document.objects.filter(user=user, is_ready=True).values('id', 'filename') 
        return Response(documents)

    def post(self, request, format=None):
        """Uploads and processes a new document using ChromaDB."""
        file_obj = request.data.get('file')
        user_id = request.data.get('user_id')
        
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'error': 'User not found.'}, status=404)

        if not file_obj:
            return Response({'error': 'No file uploaded.'}, status=400)

        document = Document.objects.create(
            user=user,
            file=file_obj,
            filename=file_obj.name,
            is_ready=False
        )
        
        raw_text = extract_text_from_file(file_obj)
        
        if not raw_text:
            document.delete()
            return Response({'error': 'Could not extract text from the file. Only PDF/TXT/DOCX supported.'}, status=400)
            
        if not index_document_in_chroma(document, raw_text):
            document.delete()
            if 'GEMINI_API_KEY' not in os.environ:
                 return Response({'error': 'Failed to index document: GEMINI_API_KEY is not configured on the server.'}, status=500)
            return Response({'error': 'Failed to index document in vector database.'}, status=500)

        return Response({
            'id': document.id, 
            'filename': document.filename, 
            'message': 'Document indexed in ChromaDB and ready for Q&A.'
        }, status=status.HTTP_201_CREATED)

def retrieve_rag_context(document_id, query_text):
    """
    Generates query embedding using Gemini and retrieves relevant text chunks 
    from ChromaDB via vector similarity search.
    """
    try:
        document = Document.objects.get(id=document_id)
        collection_name = document.collection_name
        
        if not collection_name:
             return None, None

        # Initialize Gemini Client for Embedding
        client = get_gemini_client()
        
        # Generate Query Embedding
        print(f"Generating query embedding using {EMBEDDING_MODEL}...")
        query_embedding_response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query_text],
        )
        
        query_embedding = query_embedding_response.embeddings[0].values

        # Using the query embedding to search Chroma
        chroma_client = get_chroma_client()
        collection = chroma_client.get_collection(name=collection_name)
        
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=5, 
            include=['documents']
        )
        
        context_texts = results['documents'][0] if results.get('documents') and results['documents'] and results['documents'][0] else []
        
        if not context_texts:
            return None, None

        context = "\n---\n".join(context_texts)
        
        return context, document.filename
        
    except Document.DoesNotExist:
        return None, None
    except ValueError as e:
        print(f"Gemini Client Error during retrieval: {e}")
        return None, None
    except APIError as e:
        print(f"Gemini Embedding API Error during retrieval: {e}")
        return None, None
    except Exception as e:
        print(f"Chroma retrieval or general error: {e}")
        return None, None


@csrf_exempt
@require_POST
def send_message(request, thread_id):
    """Handles the user's message, integrating RAG or using chat history."""
    try:
        data = json.loads(request.body)
        user_message = data['message']
        active_document_id = data.get('active_document_id')
        
        thread = ChatThread.objects.get(id=thread_id)

        Message.objects.create(thread=thread, sender='user', content=user_message)
        
        history = Message.objects.filter(thread=thread).exclude(sender='user', content=user_message).order_by('-created_at')[:HISTORY_LIMIT]
        contents = []
        for msg in reversed(history):
            api_role = 'model' if msg.sender == 'assistant' else 'user'
            contents.append({"role": api_role, "parts": [{"text": msg.content}]})

        is_rag_mode = False
        rag_context, doc_filename = None, None
        
        if active_document_id:
            rag_context, doc_filename = retrieve_rag_context(active_document_id, user_message)
            is_rag_mode = bool(rag_context)
            
        system_prompt = "You are a helpful and friendly assistant. Provide concise and accurate answers."
        use_google_search = not is_rag_mode
        final_user_query = user_message
        
        if is_rag_mode:
            system_prompt = (
                "You are a helpful research assistant. Your task is to answer the user's question "
                "ONLY based on the information provided in the CONTEXT section below. "
                "Respond clearly and concisely, using lists or bullet points where appropriate for readability. "
                "If the CONTEXT does not contain the answer, you MUST clearly state: 'The answer is not available in the provided document.' "
                "Do NOT use any external knowledge. Always focus on the CONTEXT provided."
            )
            
            final_user_query = (
                f"DOCUMENT SOURCE: {doc_filename}\n\n"
                f"CONTEXT:\n---\n{rag_context}\n---\n\n"
                f"USER QUESTION: {user_message}"
            )
        
        contents.append({"role": "user", "parts": [{"text": final_user_query}]})
        
        response_text = _call_gemini_api(contents, system_prompt, use_google_search)

        Message.objects.create(thread=thread, sender='assistant', content=response_text, is_rag=is_rag_mode)
        
        return JsonResponse({
            'response': response_text, 
            'is_rag': is_rag_mode,
            'doc_filename': doc_filename
        })

    except ChatThread.DoesNotExist:
        return JsonResponse({'error': 'Thread not found'}, status=404)
    except Exception as e:
        print(f"Error in send_message: {e}")
        return JsonResponse({'error': 'An internal error occurred.'}, status=500)

def _call_gemini_api(contents, system_prompt, use_google_search):
    """Handles the synchronous API call using the Google GenAI SDK."""
    try:
        client = get_gemini_client()
        
        config_params = {
            "system_instruction": system_prompt
        }
        
        if use_google_search:
            config_params["tools"] = [{"google_search": {}}]

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=config_params
        )

        return response.text if response.text else "I couldn't generate a response."
        
    except ValueError as e:
        print(f"Gemini Client Error: {e}")
        return f"Error: Failed to connect to the AI model. Details: {e}"
    except APIError as e:
        print(f"Gemini API Error: {e}")
        return f"Error: Failed to connect to the AI model. Details: {e}"
    except Exception as e:
        print(f"Unexpected Error during LLM call: {e}")
        return "Error: An unexpected internal error occurred during the AI call."

@csrf_exempt
@require_http_methods(['DELETE'])
def delete_thread(request, pk):
    """Deletes a specific chat thread."""
    try:
        thread_id = pk
        user = User.objects.first() 
        thread = ChatThread.objects.get(id=thread_id, user=user)
        thread.delete()
        return JsonResponse({'success': True, 'message': f'Thread {thread_id} deleted.'})
    except ChatThread.DoesNotExist:
        return JsonResponse({'error': 'Thread not found or user unauthorized.'}, status=404)
    except Exception as e:
        print(f"Error deleting thread: {e}")
        return JsonResponse({'error': 'An internal error occurred.'}, status=500)

@csrf_exempt
@require_http_methods(['DELETE'])
def delete_document(request, pk):
    """Deletes a specific document and its associated Chroma collection."""
    try:
        document_id = pk
        user = User.objects.first()
        document = Document.objects.get(id=document_id, user=user)
        
        collection_name = document.collection_name
        
        if collection_name:
            client = get_chroma_client()
            try:
                client.delete_collection(name=collection_name)
                print(f"Chroma collection '{collection_name}' deleted.")
            except ValueError:
                print(f"Chroma collection '{collection_name}' not found during delete (OK).")
        
        document.delete() 
        
        return JsonResponse({'success': True, 'message': f'Document {document_id} and its Chroma data deleted.'})
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found or user unauthorized.'}, status=404)
    except Exception as e:
        print(f"Error deleting document: {e}")
        return JsonResponse({'error': 'An internal error occurred.'}, status=500)
