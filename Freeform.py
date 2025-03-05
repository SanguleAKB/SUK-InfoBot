## Another method of emplementation 

from google import genai
from google.genai import types


def generate(retrieved_context,inpt):
  response_text=""
  client = genai.Client(
      vertexai=True,
      project="vertex-ai-4457270511",
      location="us-central1",
  )

  si_text1 = f"""\"You are an assistant for question-answering tasks. \"
    \"Use the following pieces of retrieved context to answer \"
    \"the question. If you don't know the answer, say that you \"
    \"don't know. Use three sentences maximum and keep the \"
    \"answer concise. Context: {retrieved_context}\""""
  model = "gemini-2.0-flash-001"
  contents = [
    types.Content(
      role="user",
      parts=[
        types.Part.from_text(text=inpt)
      ]
    )
  ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    system_instruction=[types.Part.from_text(text=si_text1)],
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    response_text += chunk.text
    return response_text

