
import openai

client = openai.OpenAI(api_key= "")  

def generate_fire_message():
    prompt = f"""화재 감지 시스템이 다음 위치에서 이상 상황을 포착했습니다: 
안전 안내방송용 문장을 정중하고 긴박하게 1문장으로 만들어줘."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content


