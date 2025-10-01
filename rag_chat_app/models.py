from django.db import models
from django.contrib.auth.models import User

class ChatThread(models.Model):
    """Represents a single conversation thread, maintaining history."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=100, default='New Chat')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Thread {self.id} - {self.title}"

class Message(models.Model):
    """Represents a single message within a thread."""
    thread = models.ForeignKey(ChatThread, on_delete=models.CASCADE)
    sender = models.CharField(max_length=10, choices=[('user', 'user'), ('assistant', 'assistant')])
    content = models.TextField()
    is_rag = models.BooleanField(default=False) 
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.content[:50]}..."
    
    class Meta:
        ordering = ['created_at']

class Document(models.Model):
    """Represents a document uploaded by the user for RAG."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    file = models.FileField(upload_to='documents/%Y/%m/%d/')
    filename = models.CharField(max_length=255)
    document_chunks = models.JSONField(default=list, blank=True)
    is_ready = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    collection_name = models.CharField(
        max_length=255, 
        unique=True, 
        null=True, 
        blank=True
    ) 
    def __str__(self):
        return self.filename
