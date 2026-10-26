import azure.functions as func
# Import the blueprint object from our apis folder
from apis.it_support import bp as it_support_blueprint

# Core Application Entry Point
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Register your blueprints here. You can stack multiple blueprints as your app grows.
app.register_blueprint(it_support_blueprint)