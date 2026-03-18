import os
import re
import json
from datetime import datetime

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

load_dotenv()


class DocumentAgent:
    """
    OCR (Document Intelligence) -> New Foundry Agent -> JSON
    Compatibile con Microsoft Azure AI Foundry "New Foundry"
    """

    def __init__(self):
        self.project_endpoint = os.getenv("PROJECT_ENDPOINT")
        self.model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")
        self.agent_name = os.getenv("AGENT_NAME", "doc-analyzer-fr-id")
        self.create_new_version = os.getenv("CREATE_NEW_AGENT_VERSION", "true").lower() == "true"

        self.doc_intelligence_endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
        self.doc_intelligence_key = os.getenv("DOC_INTELLIGENCE_KEY")

        if not self.project_endpoint:
            raise ValueError("PROJECT_ENDPOINT non trovato nel file .env")

        if not self.model_deployment:
            raise ValueError("MODEL_DEPLOYMENT_NAME non trovato nel file .env")

        self.credential = DefaultAzureCredential()

        self.project_client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=self.credential,
        )

        self.openai_client = self.project_client.get_openai_client()

        if self.doc_intelligence_endpoint and self.doc_intelligence_key:
            self.document_intelligence_client = DocumentIntelligenceClient(
                endpoint=self.doc_intelligence_endpoint,
                credential=AzureKeyCredential(self.doc_intelligence_key),
            )
            print("✅ Document Intelligence configurato")
        else:
            self.document_intelligence_client = None
            print("⚠️ Document Intelligence non configurato (OCR non disponibile)")

        self.agent = None
        if self.create_new_version:
            self.agent = self.create_agent_version()
        else:
            print(f"ℹ️ Creazione nuova versione disabilitata, uso nome agente: {self.agent_name}")

    def create_agent_version(self):
        instructions = """Tu sei un esperto nell'analisi di documenti di identità francesi.

COMPITO:
Analizza il testo estratto da una CARTE NATIONALE D'IDENTITÉ francese e restituisci i dati in formato JSON.

CAMPI DA ESTRARRE:
- nom: Cognome
- prenoms: Nome/i
- date_naissance: Data di nascita (formato DD/MM/YYYY)
- lieu_naissance: Luogo di nascita
- nationalite: Nazionalità
- numero_carte: Numero della carta
- sexe: Sesso (M/F)

REGOLE:
- Se un campo non è presente o non è leggibile, usa \"NON_PRESENTE\"
- La data deve essere nel formato DD/MM/YYYY
- sexe deve essere solo \"M\", \"F\" oppure \"NON_PRESENTE\"
- Rispondi solo con JSON valido
- Non aggiungere testo prima o dopo il JSON

FORMATO RISPOSTA:
{
  \"nom\": \"valore_o_NON_PRESENTE\",
  \"prenoms\": \"valore_o_NON_PRESENTE\",
  \"date_naissance\": \"DD/MM/YYYY_o_NON_PRESENTE\",
  \"lieu_naissance\": \"valore_o_NON_PRESENTE\",
  \"nationalite\": \"valore_o_NON_PRESENTE\",
  \"numero_carte\": \"valore_o_NON_PRESENTE\",
  \"sexe\": \"M_o_F_o_NON_PRESENTE\",
  \"confiance\": \"ALTA|MEDIA|BASSA\",
  \"statut\": \"SUCCESSO|ERRORE\"
}
"""

        print("🆕 Creazione nuova versione agente New Foundry...")

        agent = self.project_client.agents.create_version(
            agent_name=self.agent_name,
            definition=PromptAgentDefinition(
                model=self.model_deployment,
                instructions=instructions,
            ),
            description="Agente per estrazione dati da carta d'identità francese tramite OCR",
        )

        print("✅ Agente New Foundry creato")
        print(f"📝 Agent name: {agent.name}")
        print(f"🏷️ Agent version: {agent.version}")
        print(f"🆔 Agent id: {agent.id}")
        print(f"🤖 Model deployment: {self.model_deployment}")

        return agent

    def extract_text_with_ocr(self, file_path):
        if not self.document_intelligence_client:
            print("❌ Document Intelligence non configurato")
            return None

        try:
            print(f"📄 OCR extraction from: {file_path}")

            with open(file_path, "rb") as f:
                poller = self.document_intelligence_client.begin_analyze_document(
                    "prebuilt-read",
                    body=f,
                    content_type="application/pdf",
                )

            result = poller.result()
            extracted_text = result.content if result and result.content else ""

            print("✅ OCR completato")
            print(f"📝 Lunghezza testo: {len(extracted_text)} caratteri")

            return extracted_text

        except Exception as e:
            print(f"❌ OCR error: {e}")
            return None

    def analyze_with_agent(self, text):
        conversation = None

        try:
            agent_name_to_use = self.agent.name if self.agent else self.agent_name

            conversation = self.openai_client.conversations.create()
            print(f"💬 Conversation created: {conversation.id}")

            prompt = f"""Analizza questa carta d'identità francese e restituisci SOLO il JSON richiesto.

TESTO DOCUMENTO:
{text}

Restituisci SOLO JSON valido, nessun altro testo.
"""

            response = self.openai_client.responses.create(
                conversation=conversation.id,
                input=prompt,
                extra_body={
                    "agent_reference": {
                        "name": agent_name_to_use,
                        "type": "agent_reference"
                    }
                },
            )

            output_text = getattr(response, "output_text", None)

            if not output_text:
                print("⚠️ Nessun output_text trovato nella response")
                return None

            print("✅ Analisi agente completata")
            return output_text

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None

        finally:
            if conversation:
                try:
                    self.openai_client.conversations.delete(conversation_id=conversation.id)
                    print(f"🗑️ Conversation deleted: {conversation.id}")
                except Exception as cleanup_error:
                    print(f"⚠️ Impossibile cancellare la conversation: {cleanup_error}")

    def parse_json_response(self, response_text):
        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text

            result = json.loads(json_str)

            result["timestamp"] = datetime.now().isoformat()
            result["extraction_method"] = f"OCR + {self.model_deployment}"
            result["model_used"] = self.model_deployment
            result["api_type"] = "new_foundry_agents"

            if self.agent:
                result["agent_name"] = self.agent.name
                result["agent_version"] = self.agent.version
            else:
                result["agent_name"] = self.agent_name

            return result

        except Exception as e:
            return {
                "statut": "ERRORE",
                "errore": f"JSON parse error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "api_type": "new_foundry_agents",
                "raw_response": response_text[:1000] if response_text else None,
            }

    def process_document(self, file_path):
        print(f"\n🔄 Starting workflow: OCR -> New Foundry Agent -> JSON")

        extracted_text = self.extract_text_with_ocr(file_path)
        if not extracted_text:
            return {
                "statut": "ERRORE",
                "errore": "OCR failed",
                "api_type": "new_foundry_agents",
            }

        agent_response = self.analyze_with_agent(extracted_text)
        if not agent_response:
            return {
                "statut": "ERRORE",
                "errore": "Agent analysis failed",
                "api_type": "new_foundry_agents",
            }

        return self.parse_json_response(agent_response)

    def delete_created_agent_version(self):
        if not self.agent:
            print("ℹ️ Nessuna nuova versione agente da eliminare")
            return

        try:
            self.project_client.agents.delete_version(
                agent_name=self.agent.name,
                agent_version=self.agent.version,
            )
            print(f"🗑️ Agent version deleted: {self.agent.name}:{self.agent.version}")
        except Exception as e:
            print(f"⚠️ Could not delete agent version: {e}")

    def close(self):
        try:
            self.project_client.close()
        except Exception:
            pass
        try:
            self.credential.close()
        except Exception:
            pass


def main():
    pdf_path = os.getenv("PDF_PATH", "FacSimileID.pdf")
    delete_after_run = os.getenv("DELETE_CREATED_AGENT_VERSION", "false").lower() == "true"

    agent = None

    try:
        print("🚀 Starting Document Analysis Agent")
        print("📋 Mode: Azure AI Foundry New Foundry Agents")

        agent = DocumentAgent()

        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return

        result = agent.process_document(pdf_path)

        print("\n" + "=" * 60)
        print("🎯 DOCUMENT ANALYSIS RESULTS")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if delete_after_run:
            agent.delete_created_agent_version()

    except Exception as e:
        print(f"❌ Fatal error: {e}")

    finally:
        if agent:
            agent.close()


if __name__ == "__main__":
    main()
