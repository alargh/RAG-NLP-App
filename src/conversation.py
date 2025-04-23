import re
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate


def get_conversational_chain(vectors):
    llm = HuggingFaceHub(
        repo_id='HuggingFaceH4/zephyr-7b-beta',
        model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
        task="text-generation"
    )

    prompt_template = """<|system|>
    You are a helpful assistant that answers questions based on the given context.
    - Combine ALL relevant information across all chunks.
    - Only use the information provided in the context.
    - Look carefully through ALL provided context chunks before answering.
    - If a question still can't be answered from the context, reply with exactly: "I don't know."

    Context:
    {context}
    </s>
    <|user|>
    {question}
    </s>
    <|assistant|>"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    retriever = vectors.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def retrieve_and_generate(question):
        processed_question = question

        season_month_mapping = {
            "spring": ["march", "april", "may"],
            "summer": ["june", "july", "august"],
            "autumn": ["september", "october", "november"],
            "fall": ["september", "october", "november"],
            "winter": ["december", "january", "february"]
        }

        for season, months in season_month_mapping.items():
            month_terms = " ".join(months)
            pattern = rf"({season})"
            replacement = f"\\1: {month_terms},"
            processed_question = re.sub(pattern, replacement, processed_question, flags=re.IGNORECASE)

        docs = retriever.get_relevant_documents(processed_question)

        # Uncomment if debugging needed
        # print(processed_question)

        formatted_context = "\n".join([doc.page_content for doc in docs])

        # Uncomment if debugging needed
        # print("\n=== RETRIEVED CHUNKS ===")
        # for i, doc in enumerate(docs):
        #     print(f"\nCHUNK {i+1}:")
        #     print(doc.page_content)
        # print("\n=======================")

        formatted_prompt = prompt.format(context=formatted_context, question=question)

        full_response = llm(formatted_prompt)

        assistant_pattern = r'<\|assistant\|>(.*?)(?=$|<\|)'
        matches = re.findall(assistant_pattern, full_response, re.DOTALL)

        if matches:
            return matches[-1].strip()
        else:
            return full_response

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return retrieve_and_generate, memory