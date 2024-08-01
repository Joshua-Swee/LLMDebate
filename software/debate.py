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


def answer(userArgument, difficulty):
    def task():
        argument = ""
        if difficulty == "A":
            argument = ""
        if difficulty == "B":
            argument = "For your counterargument, you MUST provide justifying evidence, and you MUST explain why the evidence is relevant."
        if difficulty == "C":
            argument = "For your counterargument, you MUST provide justifying proof through a single source, and you MUST explain why that piece of evidence is relevant. For your explanation, you MUST provide backing information for it. Additionally, you MUST acknowledge of a different view to rebute your claim, to improve your credibility. Lastly, you MUST conclude your counterargument, solidifying your points."

        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful chatbot used for training debate arguments. <|eot_id|><|start_header_id|>user<|end_header_id|>

        My argument:
        {userArgument} 

        Based on my argument, pretend to be a debater and formulate a counterargument in response to my argument. 
        {argument}

        Counter argument: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
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
    n_batch=512,
    callbacks=callbacks,
    verbose=True,
    stop=["<|end_of_text|>", "<|eot_id|>", "]]>"]
)

with gr.Blocks(title="Debate Agent", css="footer{display:none !important}") as demo:
    gr.Markdown("# Debate Agent")
    chatbot = gr.Chatbot()
    difficultyCheckbox = gr.Radio(
        ["A", "B", "C"],
        value="A",
        label="Answer Style",
        info="How would you like me to debate?")
    msg = gr.Textbox(placeholder="Argue with me...", info="Hit 'Enter' to submit your argument!")
    clearBtn = gr.Button("Clear")


    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, difficulty):
        userArgument = history[-1][0]
        history[-1][1] = ""
        answer(userArgument=userArgument, difficulty=difficulty)
        while True:
          try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
              break
            history[-1][1] += next_token
            yield history
          except Empty:
            continue

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot, difficultyCheckbox], chatbot)
    clearBtn.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()