
from openai import OpenAI


def test_stream():
    # Initialize OpenAI client
    client = OpenAI(
        base_url="http://localhost:8080/v1/",
        api_key="1b509260-bdfc-4628-9bae-dd9814e902a1",
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    print(messages)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct", 
        messages=messages, # type: ignore 
        stream=True
    ) # type: ignore

    content = ""
    for chunk in response:
        print(chunk)
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content)
            content += chunk.choices[0].delta.content

    print(content)
    if content:
        print("Final response: ", content)
    else:
        raise Exception("No response received: ", content)


def test_non_stream():
    # Initialize OpenAI client
    client = OpenAI(
        base_url="http://localhost:8080/v1/",
        api_key="1b509260-bdfc-4628-9bae-dd9814e902a1",
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    print(messages)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct", 
        messages=messages, # type: ignore 
        stream=False
    ) # type: ignore
    print(response)
    # raise Exception(f"Response: {response}")


if __name__ == "__main__":
    test_stream()
    test_non_stream()
