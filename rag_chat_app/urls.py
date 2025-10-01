from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'), 
    path('api/thread/create/', views.create_thread, name='api_create_thread'),
    path('api/thread/<int:thread_id>/messages/', views.get_messages, name='api_get_messages'),
    path('api/thread/<int:thread_id>/send_message/', views.send_message, name='api_send_message'),
    path('api/documents/', views.DocumentListCreateView.as_view(), name='api_document_list_create'),
    path('api/thread/<int:pk>/', views.delete_thread, name='thread-delete'),
    path('api/documents/<int:pk>/', views.delete_document, name='document-delete'),
]
