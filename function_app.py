import azure.functions as func
from apis.it_support import bp as it_support_blueprint

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
app.register_blueprint(it_support_blueprint)