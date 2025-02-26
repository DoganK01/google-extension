from PIL import Image as PILImage
from io import BytesIO
from langchain_core.runnables.graph import MermaidDrawMethod

def get_graph(graph):
    img_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    image = PILImage.open(BytesIO(img_data))
    image.show()
