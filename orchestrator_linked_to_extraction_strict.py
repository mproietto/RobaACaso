import os
import traceback

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

load_dotenv()


def main():
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    if not project_endpoint:
        raise ValueError("PROJECT_ENDPOINT not found in .env")

    if not model_deployment:
        raise ValueError("MODEL_DEPLOYMENT_NAME not found in .env")

    credential = DefaultAzureCredential()
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
    )

    function_definition = {
        "type": "function",
        "function": {
            "name": "analyze_french_id_document",
            "description": "MANDATORY function to analyze French ID documents. Must ALWAYS be called for any document analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_text": {
                        "type": "string",
                        "description": "Complete text from French ID document"
                    }
                },
                "required": ["document_text"]
            }
        }
    }

    ultra_strict_instructions = """Tu es un agent orchestrateur qui DOIT OBLIGATOIREMENT utiliser la fonction analyze_french_id_document.

⚠️ RÈGLE ABSOLUE: Tu ne peux PAS analyser de documents toi-même. Tu DOIS TOUJOURS appeler analyze_french_id_document.

📋 PROCESSUS OBLIGATOIRE:
1. Recevoir un texte de document d'identité français
2. Appeler IMMÉDIATEMENT analyze_french_id_document(document_text=texte_complet)
3. Retourner EXACTEMENT le résultat de cette fonction

🚫 INTERDICTIONS STRICTES:
- Ne JAMAIS extraire d'informations toi-même
- Ne JAMAIS créer de JSON toi-même
- Ne JAMAIS répondre sans appeler la fonction
- Ne JAMAIS analyser le contenu du document

✅ SEULE ACTION AUTORISÉE:
Si tu reçois un texte de document → appelle analyze_french_id_document(document_text=texte)

Tu DOIS utiliser la fonction analyze_french_id_document pour TOUT texte de document.
"""

    try:
        new_orchestrator = project_client.agents.create_version(
            agent_name="OrchestratorLinkedToExtractionSTRICT",
            definition=PromptAgentDefinition(
                model=model_deployment,
                instructions=ultra_strict_instructions,
                tools=[function_definition],
            ),
            description="Orchestrator that MUST use function calls - NO exceptions",
        )

        print("✅ NEW ULTRA-STRICT ORCHESTRATOR CREATED!")
        print(f"📝 Agent name: {new_orchestrator.name}")
        print(f"🏷️ Agent version: {new_orchestrator.version}")
        print(f"🆔 Agent id: {new_orchestrator.id}")
        print(f"🤖 Model deployment: {model_deployment}")

        tools = getattr(new_orchestrator, "tools", None)
        if tools is not None:
            print(f"🔧 Tools count: {len(tools)}")
        else:
            print("🔧 Tools count: not exposed by response object")

    except Exception as e:
        print(f"❌ Error creating new orchestrator: {e}")
        traceback.print_exc()

    finally:
        try:
            project_client.close()
        except Exception:
            pass

        try:
            credential.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
