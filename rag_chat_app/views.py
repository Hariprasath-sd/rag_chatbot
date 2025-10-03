import json
import os
import uuid
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_http_methods
from django.contrib.auth.models import User
# NEW: Import authentication tools
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required 
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt # Keep for APIView POST (Document upload)

from google import genai 
from google.genai.errors import APIError 

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document as DocxDocument

<<<<<<< HEAD
# --- Configuration Constants ---
=======
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986
EMBEDDING_MODEL = "gemini-embedding-001" 
LLM_MODEL = "gemini-2.5-flash"
HISTORY_LIMIT = 5 
CHROMA_PATH = "./chroma_db" 
# -------------------------------

# --- Placeholder Models (Kept for environment stability) ---
try:
    from .models import ChatThread, Message, Document
except ImportError:
    # ... (Placeholder models are omitted for brevity, assume they are correctly defined)
    class User:
        def __init__(self): self.id = 1
        @staticmethod
        def objects(): 
            class Manager:
                def create_user(self, **kwargs): return User()
                def first(self): return User()
                def get(self, **kwargs): return User()
                def filter(self, **kwargs): return []
                def exists(self): return False
            return Manager()

<<<<<<< HEAD
# --- Authentication Views (NEW) ---

def register_view(request):
    """Handles user registration."""
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        
        if password != password_confirm:
            return render(request, 'rag_chat_app/register.html', {'error': 'Passwords do not match.'})

        if User.objects.filter(username=username).exists():
            return render(request, 'rag_chat_app/register.html', {'error': 'Username already taken.'})

        try:
            user = User.objects.create_user(username=username, password=password)
            user.save()
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('index')
            else:
                return render(request, 'rag_chat_app/register.html', {'error': 'Registration successful, but auto-login failed.'})

        except Exception as e:
            return render(request, 'rag_chat_app/register.html', {'error': f'An error occurred: {e}'})

    return render(request, 'rag_chat_app/register.html')

def login_view(request):
    """Handles user login."""
    if request.user.is_authenticated:
        return redirect('index')
        
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            return render(request, 'rag_chat_app/login.html', {'error': 'Invalid username or password.'})

    return render(request, 'rag_chat_app/login.html')

@login_required 
def logout_view(request):
    """Logs the user out and redirects to the login page."""
    logout(request)
    return redirect('login')

# --- Utility Functions (unchanged) ---

def get_gemini_client():
    # ... (content remains the same)
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Cannot initialize Gemini client.")
    return genai.Client(api_key=api_key)

def get_chroma_client():
    # ... (content remains the same)
=======
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
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986
    return chromadb.PersistentClient(path=CHROMA_PATH)

def extract_text_from_file(file_obj) -> str:
    # ... (content remains the same)
    text = ""
    file_name = file_obj.name.lower()
    # ... (PDF/DOCX/TXT extraction logic)
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
<<<<<<< HEAD
    # ... (content remains the same)
    collection_name = f"doc-{document_obj.id}-{uuid.uuid4().hex[:8]}"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300, length_function=len, is_separator_regex=False)
    texts = text_splitter.split_text(raw_text)
    
    if not texts: return False
    try:
        client = get_gemini_client() 
        embedding_response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
        embeddings = [e.values for e in embedding_response.embeddings] 
        if len(embeddings) != len(texts): return False

=======
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
            
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986
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
<<<<<<< HEAD
        print(f"Indexing Error: {e}")
=======
        print(f"Chroma indexing or general error: {e}")
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986
        return False


# --- Core Application Views (Secured) ---

@login_required # SECURED: Only accessible to logged-in users
def index(request):
    """Renders the single-page chat interface."""
    user = request.user 
    threads = ChatThread.objects.filter(user=user).order_by('-created_at')
    return render(request, 'rag_chat_app/index.html', {'user_id': user.id, 'threads': threads, 'username': user.username})

# Removed @csrf_exempt
@require_POST
@login_required # SECURED: Only logged-in users can create threads
def create_thread(request):
    """Creates a new chat thread for the authenticated user."""
    try:
        user = request.user # Use the authenticated user
        thread = ChatThread.objects.create(user=user, title="New Chat")
        
        Message.objects.create(
            thread=thread, 
            sender='assistant', 
            content='Welcome! Upload a document to ask grounded questions, or start a general chat.'
        )
        return JsonResponse({'thread_id': thread.id, 'title': thread.title})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required # SECURED
def get_messages(request, thread_id):
    """Retrieves all messages for a given thread, ensuring it belongs to the user."""
    try:
        # Ensure the thread belongs to the authenticated user
        thread = ChatThread.objects.get(id=thread_id, user=request.user)
        messages = Message.objects.filter(thread=thread).values('sender', 'content', 'is_rag')
        return JsonResponse({'messages': list(messages), 'title': thread.title})
    except ChatThread.DoesNotExist:
        return JsonResponse({'error': 'Thread not found or user unauthorized.'}, status=404)

class DocumentListCreateView(APIView):
    """Handles document upload and listing."""
    # Note: APIView's .post method will require a CSRF token unless exempted, 
    # which is handled by the client-side fetch logic (X-CSRFToken header).

    def get(self, request, format=None):
        """List all ready documents for the current user."""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required.'}, status=status.HTTP_403_FORBIDDEN)
            
        documents = Document.objects.filter(user=request.user, is_ready=True).values('id', 'filename') 
        return Response(documents)

    def post(self, request, format=None):
        """Uploads and processes a new document using ChromaDB."""
        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required.'}, status=status.HTTP_403_FORBIDDEN)
            
        file_obj = request.data.get('file')
        user = request.user 

        if not file_obj:
            return Response({'error': 'No file uploaded.'}, status=status.HTTP_400_BAD_REQUEST)

        document = Document.objects.create(
            user=user,
            # file=file_obj, # Assuming `file` field is handled by your actual model
            filename=file_obj.name,
            is_ready=False
        )
        
        raw_text = extract_text_from_file(file_obj)
        
        if not raw_text:
            document.delete()
            return Response({'error': 'Could not extract text from the file. Only PDF/TXT/DOCX supported.'}, status=status.HTTP_400_BAD_REQUEST)
            
        if not index_document_in_chroma(document, raw_text):
            document.delete()
            if 'GEMINI_API_KEY' not in os.environ:
<<<<<<< HEAD
                 return Response({'error': 'Failed to index document: GEMINI_API_KEY is not configured on the server.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response({'error': 'Failed to index document in vector database.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
=======
                 return Response({'error': 'Failed to index document: GEMINI_API_KEY is not configured on the server.'}, status=500)
            return Response({'error': 'Failed to index document in vector database.'}, status=500)
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986

        return Response({
            'id': document.id, 
            'filename': document.filename, 
            'message': 'Document indexed in ChromaDB and ready for Q&A.'
        }, status=status.HTTP_201_CREATED)

def retrieve_rag_context(document_id, query_text):
<<<<<<< HEAD
    # ... (content remains the same - Retrieval logic)
=======
    """
    Generates query embedding using Gemini and retrieves relevant text chunks 
    from ChromaDB via vector similarity search.
    """
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986
    try:
        document = Document.objects.get(id=document_id)
        collection_name = document.collection_name
        
        if not collection_name: return None, None

<<<<<<< HEAD
        client = get_gemini_client()
        query_embedding_response = client.models.embed_content(model=EMBEDDING_MODEL, contents=[query_text])
        query_embedding = query_embedding_response.embeddings[0].values

        chroma_client = get_chroma_client()
        collection = chroma_client.get_collection(name=collection_name)
=======
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
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986
        
        results = collection.query(query_embeddings=[query_embedding], n_results=5, include=['documents'])
        context_texts = results['documents'][0] if results.get('documents') and results['documents'] and results['documents'][0] else []
        
        if not context_texts: return None, None

        context = "\n---\n".join(context_texts)
        return context, document.filename
        
<<<<<<< HEAD
    except Exception as e:
        print(f"Retrieval Error: {e}")
=======
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
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986
        return None, None


# Removed @csrf_exempt
@require_POST
@login_required # SECURED
def send_message(request, thread_id):
    """Handles the user's message, integrating RAG or using chat history."""
    try:
        data = json.loads(request.body)
        user_message = data['message']
        active_document_id = data.get('active_document_id')
        
        thread = ChatThread.objects.get(id=thread_id, user=request.user) # Check ownership

        Message.objects.create(thread=thread, sender='user', content=user_message)
        
        # Prepare chat history (omitted for brevity)
        history = Message.objects.filter(thread=thread).exclude(sender='user', content=user_message).order_by('-created_at')[:HISTORY_LIMIT]
        contents = []
        for msg in reversed(history):
            api_role = 'model' if msg.sender == 'assistant' else 'user'
            contents.append({"role": api_role, "parts": [{"text": msg.content}]})

        is_rag_mode = False
        rag_context, doc_filename = None, None
        
        if active_document_id:
            try:
                document = Document.objects.get(id=active_document_id, user=request.user) # Check ownership
                rag_context, doc_filename = retrieve_rag_context(active_document_id, user_message)
                is_rag_mode = bool(rag_context)
            except Document.DoesNotExist:
                 is_rag_mode = False 
            
        system_prompt = "You are a helpful and friendly assistant. Provide concise and accurate answers."
        use_google_search = not is_rag_mode
        final_user_query = user_message
        
        if is_rag_mode:
            # --- RAG TUNING FIX APPLIED HERE ---
            system_prompt = (
                "You are a helpful research assistant. Your task is to answer the user's question "
                "ONLY based on the information provided in the CONTEXT section below. "
                "Respond clearly and concisely, using lists or bullet points where appropriate for readability. "
                "If the CONTEXT does not contain the answer, acknowledge the query but gently state that the specific information is not present in the document. For example: 'Based on the provided document, the specific name of the candidate is not mentioned.' Do NOT make up information."
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
        return JsonResponse({'error': 'Thread not found or user unauthorized.'}, status=404)
    except Exception as e:
        print(f"Error in send_message: {e}")
        return JsonResponse({'error': 'An internal error occurred.'}, status=500)

def _call_gemini_api(contents, system_prompt, use_google_search):
    # ... (content remains the same - LLM call logic)
    try:
        client = get_gemini_client()
<<<<<<< HEAD
        config_params = {"system_instruction": system_prompt}
=======
        
        config_params = {
            "system_instruction": system_prompt
        }
        
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986
        if use_google_search:
            config_params["tools"] = [{"google_search": {}}]

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config=config_params
        )
<<<<<<< HEAD
        return response.text if response.text else "I couldn't generate a response."
    except Exception as e:
        print(f"LLM Call Error: {e}")
        return f"Error: Failed to connect to the AI model. Details: {e}"
=======

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
>>>>>>> 6fcbea783ef614ff42be29c73e618348bc621986


# Removed @csrf_exempt
@require_http_methods(['DELETE'])
@login_required # SECURED
def delete_thread(request, pk):
    """Deletes a specific chat thread for the authenticated user."""
    try:
        thread_id = pk
        thread = ChatThread.objects.get(id=thread_id, user=request.user) # Check ownership
        thread.delete()
        return JsonResponse({'success': True, 'message': f'Thread {thread_id} deleted.'})
    except ChatThread.DoesNotExist:
        return JsonResponse({'error': 'Thread not found or user unauthorized.'}, status=404)
    except Exception as e:
        print(f"Error deleting thread: {e}")
        return JsonResponse({'error': 'An internal error occurred.'}, status=500)

# Removed @csrf_exempt
@require_http_methods(['DELETE'])
@login_required # SECURED
def delete_document(request, pk):
    """Deletes a specific document and its associated Chroma collection for the authenticated user."""
    try:
        document_id = pk
        document = Document.objects.get(id=document_id, user=request.user) # Check ownership
        
        collection_name = document.collection_name
        
        if collection_name:
            client = get_chroma_client()
            try:
                client.delete_collection(name=collection_name)
            except ValueError:
                pass # Collection not found is fine
        
        document.delete() 
        
        return JsonResponse({'success': True, 'message': f'Document {document_id} and its Chroma data deleted.'})
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found or user unauthorized.'}, status=404)
    except Exception as e:
        print(f"Error deleting document: {e}")
        return JsonResponse({'error': 'An internal error occurred.'}, status=500)