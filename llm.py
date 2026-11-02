
import requests
from config import settings


def appeler_llm(
    system: str,
    user: str,
    temperature: float = 0.2
) -> str:
    reponse = requests.post(
        settings.llm_url,
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}"
        },
        json={
            "model": settings.llm_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=60,
    )

    reponse.raise_for_status()
    return reponse.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(
        appeler_llm(
            "Tu réponds en français, en deux phrases maximum.",
            "Explique ce qu'est un token."
        )
    )
