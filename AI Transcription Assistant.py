import gradio as gr
from transformers import pipeline

# Initialize the pipelines once to avoid re-initialization overhead
transcription_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
)

qa_pipe = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

def process_audio_and_question(audio_file, question):

    transcription = transcription_pipe(audio_file, batch_size=8)['text']    
    response = qa_pipe(question=question, context=transcription)['answer']
    
    return transcription, response

with gr.Blocks() as demo:
    gr.Markdown("# Audio Transcription and QA")
    gr.Markdown("Upload an audio file and ask questions about its transcription.")
    
    with gr.Row():
        audio_input = gr.Audio(type='filepath')
        question_input = gr.Textbox(placeholder='Ask a question about the transcription...')
        
    with gr.Row():
        output_transcription = gr.Textbox(label='Transcription', placeholder='Transcription will appear here...')
        output_response = gr.Textbox(label='Answer', placeholder='Answer will appear here...')
        
    submit_btn = gr.Button("Submit")
    
    submit_btn.click(
        process_audio_and_question,
        inputs=[audio_input, question_input],
        outputs=[output_transcription, output_response]
    )

demo.launch(share=True, inbrowser=True)
