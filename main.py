
import logging

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from schema import ChatRequest

from mistralai.client import Mistral
from pydantic import ValidationError

from config import Settings
from schema import GeneratedTextResponse, Message


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mon Assistant IA",
    description="Assistant intelligent utilisant les modèles Mistral.",
    version="1.0.0"
)

settings = Settings()
client = Mistral(api_key=settings.LLM_API_KEY)


# --------------------------------------------------
# MODELES DISPONIBLES
# --------------------------------------------------

AVAILABLE_MODELS = [
    {
        "id": "ministral-8b-latest",
        "name": "Ministral 8B"
    },
    {
        "id": "mistral-small-latest",
        "name": "Mistral Small"
    },
    {
        "id": "mistral-large-latest",
        "name": "Mistral Large"
    }
]

DEFAULT_MODEL = "ministral-8b-latest"

ALLOWED_MODELS = {
    model["id"] for model in AVAILABLE_MODELS
}


# --------------------------------------------------
# CONFIGURATION FRONTEND
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(directory="templates")


# --------------------------------------------------
# PROMPTS SYSTEME
# --------------------------------------------------

QUESTION_SYSTEM_PROMPT = """
Tu es un assistant intelligent, amical et pédagogique.

Tu te présentes comme un assistant IA.
Réponds clairement et précisément aux questions.
Utilise la langue de l'utilisateur.
Sois concis et facile à comprendre.
"""

GENERATE_SYSTEM_PROMPT = """
Tu es un assistant utile et pédagogique.

Réponds dans la langue de l'utilisateur.

Retourne uniquement un objet JSON valide avec exactement
les trois clés suivantes :

{
    "sujet": "texte",
    "explication": "texte",
    "utilisation": [
        "exemple 1",
        "exemple 2"
    ]
}

Tous les champs sont obligatoires et ne doivent pas être vides.
La valeur de explication doit être une chaîne de caractères.
Chaque élément de utilisation doit être une chaîne de caractères.
"""


# --------------------------------------------------
# FONCTION COMMUNE POUR MISTRAL
# --------------------------------------------------

def call_mistral(
    prompt: str,
    model: str,
    mode: str
):
    """
    Envoie une requête à Mistral.

    mode = question ou generate
    """

    if mode == "question":
        system_prompt = QUESTION_SYSTEM_PROMPT
        temperature = 0.5
        response_format = {"type": "text"}

    else:
        system_prompt = GENERATE_SYSTEM_PROMPT
        temperature = 0.7
        response_format = {"type": "json_object"}

    response = client.chat.complete(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=response_format
    )

    if not response.choices:
        raise ValueError("Mistral n'a retourné aucun résultat.")

    contenu = response.choices[0].message.content

    if not isinstance(contenu, str) or not contenu.strip():
        raise ValueError("La réponse du modèle est vide ou invalide.")

    return contenu


# --------------------------------------------------
# ROUTE PAGE D'ACCUEIL
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "models": AVAILABLE_MODELS
        }
    )


# --------------------------------------------------
# ROUTE MODELES
# --------------------------------------------------

@app.get("/models")
async def get_models():

    return {
        "models": AVAILABLE_MODELS
    }



# --------------------------------------------------
# ROUTE CHAT
# --------------------------------------------------

@app.post("/chat")
async def chat(request: ChatRequest):

    # Vérifier le modèle demandé
    if request.model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="Modèle IA non autorisé."
        )

    # Vérifier les rôles des messages
    conversation = []

    for message in request.messages:
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail="Rôle de message invalide."
            )

        conversation.append({
            "role": message.role,
            "content": message.content
        })

    # Ajouter les instructions de l'assistant
    mistral_messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant IA pédagogique. "
                "Réponds en français, simplement et clairement. "
                "Tiens compte des messages précédents."
            )
        },
        *conversation
    ]

    try:
        response = await run_in_threadpool(
            client.chat.complete,
            model=request.model,
            messages=mistral_messages,
            temperature=0.7
        )

        reply = response.choices[0].message.content

        if not isinstance(reply, str) or not reply.strip():
            raise HTTPException(
                status_code=502,
                detail="Mistral n'a pas fourni de réponse valide."
            )

        return {"reply": reply}

    except HTTPException:
        raise

    except Exception:
        logger.exception("Erreur pendant la conversation avec Mistral")

        raise HTTPException(
            status_code=502,
            detail="Impossible d'obtenir une réponse de Mistral."
        )

    
# --------------------------------------------------
# ROUTE ASSISTANT HTML
# --------------------------------------------------

@app.post("/assistant", response_class=HTMLResponse)
async def assistant(
    request: Request,
    prompt: str = Form(...),
    model: str = Form(DEFAULT_MODEL),
    mode: str = Form("question")
):

    prompt = prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Le prompt ne peut pas être vide."
        )

    if model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="Modèle invalide."
        )

    if mode not in ["question", "generate"]:
        raise HTTPException(
            status_code=400,
            detail="Mode invalide."
        )

    try:

        contenu = await run_in_threadpool(
            call_mistral,
            prompt,
            model,
            mode
        )

        if mode == "question":

            context = {
                "mode": "question",
                "prompt": prompt,
                "answer": contenu,
                "model": model
            }

        else:

            data = GeneratedTextResponse.model_validate_json(
                contenu
            )

            context = {
                "mode": "generate",
                "prompt": prompt,
                "data": data.model_dump(),
                "model": model
            }

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context=context
        )

    except ValidationError:
        logger.exception("Erreur de validation de la réponse JSON.")

        error_message = "Le modèle a retourné des données invalides."

    except Exception:
        logger.exception("Erreur pendant la génération Mistral.")

        error_message = "Une erreur est survenue pendant la génération."

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "error": error_message,
            "prompt": prompt,
            "model": model
        },
        status_code=502
    )


# --------------------------------------------------
# ROUTE API GENERATE JSON
# --------------------------------------------------

@app.post(
    "/generate",
    response_model=GeneratedTextResponse
)
async def generate_text(prompt: Message):

    texte = prompt.prompt.strip()

    if not texte:
        raise HTTPException(
            status_code=400,
            detail="Le prompt ne peut pas être vide."
        )

    try:

        contenu = await run_in_threadpool(
            call_mistral,
            texte,
            DEFAULT_MODEL,
            "generate"
        )

        return GeneratedTextResponse.model_validate_json(
            contenu
        )

    except ValidationError as exc:

        logger.exception("Réponse JSON non conforme.")

        raise HTTPException(
            status_code=502,
            detail="Le modèle a retourné un JSON non conforme."
        ) from exc

    except Exception as exc:

        logger.exception("Erreur pendant l'appel Mistral.")

        raise HTTPException(
            status_code=502,
            detail="Erreur lors de la génération auprès du modèle."
        ) from exc
