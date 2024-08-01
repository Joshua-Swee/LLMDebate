import gradio as gr
from typing import Any
from queue import Queue, Empty
from langchain.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread

q = Queue()
job_done = object()

class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()


def answer(argumentA, argumentB):

    def task():

        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful chatbot used for comparing debate arguments. <|eot_id|><|start_header_id|>user<|end_header_id|>

        Argument A: "{argumentA}" 
        ---

        Argument B: "{argumentB}"
        ---
        
        Evaluate which argument is better, based on how detailed is their capability to consider evidence, reasoning and potential counterpoints.

        Evaluation: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        response = llm(template)
        q.put(job_done)

    t = Thread(target=task)
    t.start()


callbacks = [QueueCallback(q)]

llm = LlamaCpp(
    model_path="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    max_tokens=-1,
    n_ctx=15000,
    n_batch=1024,
    callbacks=callbacks,
    verbose=True,
    stop=["<|end_of_text|>", "<|eot_id|>", "]]>"]
)

with gr.Blocks(title="Judge Agent", css="footer{display:none !important}") as demo:
    gr.Markdown("# Judge Agent")
    chatbot = gr.Chatbot()
    argumentAtextbox = gr.Textbox(label="Argument A", placeholder="Enter your first argument:")
    argumentBtextbox = gr.Textbox(label="Argument B", placeholder="Enter your second argument:")
    sendBtn = gr.Button("Submit")
    clearBtn = gr.Button("Clear")


    def user(argumenta, argumentb, history):
        return "", "", history + [[f"Argument A: {argumenta}", None]] + [[f"Argument B: {argumentb}", None]]

    def bot(history):
        argumentA = history[-2][0]
        argumentB = history[-1][0]

        history[-1][1] = ""
        answer(argumentA, argumentB)
        while True:
          try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
              break
            history[-1][1] += next_token
            yield history
          except Empty:
            continue
    sendBtn.click(user, [argumentAtextbox, argumentBtextbox, chatbot], [argumentAtextbox, argumentBtextbox, chatbot], queue=False).then(bot, [chatbot], chatbot)
    clearBtn.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()