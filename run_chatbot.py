from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import gradio as gr

# 3. Load and use vector database
def load_vector_database(model, device, database_path):
    model_name = model
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True} # set True to compute cosine simi
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = Chroma( embedding_function=hf, persist_directory=database_path)
    
    return db

# 4. Utility function
def docs_to_dict(docs,query,company):
    d={"question":query,
        "Company":company,
        }
    for i,(doc,score) in enumerate(docs):
        #d[f"page{i}_year"] = doc.metadata['year']
        d[f"page{i}"] = doc.page_content
        d[f"page{i}_Q-K"] = doc.metadata['Q-K']
    
    return d

# 5. Testing chatbot
def finchatv2(question,history, company):
    query= question
    docs = retriever.similarity_search_with_relevance_scores(query, k=6, filter={"company": company})
    d = docs_to_dict(docs,query,company)
    tollm=example_prompt.format(**d)
    res = llm.invoke(tollm)
    metainformation = docs_to_info(docs)
    bot_message = "## Answer about "+company+"\n**"+res.content+"**\n ### Source
    history.append((question, bot_message))
    return "", history

# 6. Interface
def interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(height=700)
        msg = gr.Textbox(label="Question")
        clear = gr.ClearButton([msg, chatbot])
        companydrop = gr.Dropdown(
                # Here is the list of companies from which we have data
                ["AAPL", "LULU", "BABA", "QSR", "JPM", "RY"],
                value='AAPL',
                multiselect=False,
                )
        examples = gr.Examples(examples=["China growth",
                                        "Services revenue",
                                        "Stock based compensation"],
                                        inputs=[msg])
        msg.submit(finchatv2,
                    inputs=[msg, chatbot,companydrop],
                    outputs=[msg,chatbot])
    demo.queue()
    demo.launch(share=True,debug=True)

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse arguments for the financial chatbot.")
    
    # Add an optional argument with a default value
    parser.add_argument('--google-key', type=str, default='',
                        help='Specify your created PdfGeneratorApi')
    parser.add_argument('--llm', type=str, default='gemini-2.5-flash',
                       help='Specify the LLM')
    parser.add_argument('--encoder', type=str, default='BAAI/bge-small-en',
                       help='Specify the retriever model')
    parser.add_argument('--database-path', type=str, default='./db/')

    args = parser.parse_args()

    device = torch.device('cuda:0')

    # 1. Using Gemini LLM
    os.environ["GOOGLE_API_KEY"] = args.google_key
    llm = ChatGoogleGenerativeAI(model=args.llm)

    # 2. Prompt Construction
    example_prompt = PromptTemplate(
      input_variables=["question", "Company",
          "page0", "page1", "page2","page3", "page4",
          "page0_Q-K","page1_Q-K","page2_Q-K",
          "page3_Q-K","page4_Q-K",
      ],
      template="""
          You are a financial expert.
          Using only the information provided below, answer the question.
          # Financial documents for {Company}
          Below are documents with information related to financial statements.
          ## Document 1 , Type={page0_Q-K}:
          {page0}
          ## Document 2 , Type={page1_Q-K}:
          {page1}
          ## Document 3 , Type={page2_Q-K}:
          {page2}
          ## Document 4 , Type={page3_Q-K}:
          {page3}
          ## Document 5 , Type={page4_Q-K}:
          {page4}
          )
          ## Question
          The question we are trying to answer is: {question}
          ##Answer:
          """
    )

    db = load_vector_database(args.encoder, device, args.database_path)
    interface()   