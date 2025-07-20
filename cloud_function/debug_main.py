import os
import functions_framework

@functions_framework.http
def debug_api_key(request):
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        return "GEMINI_API_KEY environment variable is set.", 200
    else:
        return "GEMINI_API_KEY environment variable is NOT set.", 500
