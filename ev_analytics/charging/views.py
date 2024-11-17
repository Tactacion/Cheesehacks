from .analytics import EVAnalytics
# In your Django view
def dashboard_view(request):
    return render(request, 'dashboard.html')
