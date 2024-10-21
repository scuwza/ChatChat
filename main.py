from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from models.GLM_4 import ChatGLM4_LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.vector_store import initialize_faiss

import warnings

warnings.filterwarnings('ignore')


def create_prompt_template():
    template = (
        "【指令】根据已知信息，简洁和专业的来回答问题。"
        "如果无法从中得到答案，请自由发挥，答案请使用中文。\n\n"
        "【已知信息】{context}\n\n"
        "【问题】{question}\n"
    )
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


def create_memory():
    return ConversationBufferMemory(memory_key="chat_history", input_key="question")


def main():
    model_path = "/root/autodl-tmp/glm-4-9b-chat"
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

    llm = ChatGLM4_LLM(model_name_or_path=model_path, gen_kwargs=gen_kwargs)
    memory = create_memory()
    prompt = create_prompt_template()

    # llm_chain = LLMChain(
    #     llm=llm,
    #     prompt=prompt_template,
    #     memory=memory
    # )
    #
    # while True:
    #     question = input("\nYou: ")
    #     if question.lower() in ["exit", "quit","q"]:
    #         break
    #
    #     context = ""  # replace with your context-fetching logic if needed
    #     response = llm_chain.run(context=context, question=question)
    #     print("GLM-4:", response)

    faiss_db = initialize_faiss()

    while True:
        question = input("\nYou: ")
        if question.lower() in ["exit", "quit","q"]:
            break

        # 执行相似度搜索以找到与查询相似的嵌入
        similar_embeddings = faiss_db.similarity_search(question)
        # 创建检索器和链
        retriever = similar_embeddings.as_retriever()
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 执行RAG链以获得响应
        response = rag_chain.invoke(query)
        print("GLM-4:", response)



if __name__ == "__main__":
    main()
