import azure.functions as func
from apis.knowledge_base import bp as knowledge_base_blueprint

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
app.register_blueprint(knowledge_base_blueprint)