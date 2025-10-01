from django.contrib import admin
from .models import ChatThread, Message, Document

@admin.register(ChatThread)
class ChatThreadAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'created_at')
    list_filter = ('created_at', 'user')
    search_fields = ('title', 'user__username')
    
    class MessageInline(admin.TabularInline):
        model = Message
        extra = 0
        fields = ('sender', 'content', 'is_rag', 'created_at')
        readonly_fields = ('created_at',)

    inlines = [MessageInline]


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('id', 'filename', 'user', 'is_ready', 'created_at')
    list_filter = ('is_ready', 'created_at', 'user')
    search_fields = ('filename', 'user__username')
    readonly_fields = ('document_chunks', 'created_at') 
