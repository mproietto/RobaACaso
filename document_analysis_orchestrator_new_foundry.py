import os
import json
import re
from datetime import datetime

from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

load_dotenv()


class DocumentAnalysisOrchestrator:
    def __init__(self):
        self.orchestrator_agent_name = os.getenv(
            "ORCHESTRATOR_AGENT_NAME",
            "OrchestratorLinkedToExtraction"
        )
        self.specialist_agent_name = os.getenv(
            "SPECIALIST_AGENT_NAME",
            "doc-analyzer-fr-id"
        )

        self.project_endpoint = os.getenv("PROJECT_ENDPOINT")
        self.model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME", "UNKNOWN_MODEL")
        self.doc_intelligence_endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
        self.doc_intelligence_key = os.getenv("DOC_INTELLIGENCE_KEY")

        self.credential = None
        self.project_client = None
        self.openai_client = None
        self.doc_client = None

        self._connect()

    def _connect(self):
        if not self.project_endpoint:
            raise ValueError(
                "La variabile d'ambiente PROJECT_ENDPOINT non è impostata. Controlla il tuo file .env."
            )
        if not self.doc_intelligence_endpoint:
            raise ValueError(
                "La variabile d'ambiente DOC_INTELLIGENCE_ENDPOINT non è impostata. Controlla il tuo file .env."
            )
        if not self.doc_intelligence_key:
            raise ValueError(
                "La variabile d'ambiente DOC_INTELLIGENCE_KEY non è impostata. Controlla il tuo file .env."
            )

        self.credential = DefaultAzureCredential()

        self.project_client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=self.credential,
        )

        self.openai_client = self.project_client.get_openai_client()

        self.doc_client = DocumentIntelligenceClient(
            endpoint=self.doc_intelligence_endpoint,
            credential=AzureKeyCredential(self.doc_intelligence_key),
        )

    def extract_text_from_pdf(self, file_path):
        try:
            print(f"📄 OCR extraction from: {file_path}")

            with open(file_path, "rb") as f:
                poller = self.doc_client.begin_analyze_document(
                    "prebuilt-read",
                    body=f,
                    content_type="application/pdf",
                )

            result = poller.result()
            extracted_text = result.content if result and result.content else ""

            print("✅ OCR completed")
            print(f"📝 Text length: {len(extracted_text)} characters")

            return extracted_text

        except Exception as e:
            print(f"❌ OCR error: {e}")
            return None

    def _extract_output_text(self, response):
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        collected = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue

            for content in getattr(item, "content", []) or []:
                content_type = getattr(content, "type", None)

                if content_type == "output_text":
                    text_value = getattr(content, "text", None)
                    if isinstance(text_value, str):
                        collected.append(text_value)
                    else:
                        nested_value = getattr(text_value, "value", None) if text_value else None
                        if nested_value:
                            collected.append(nested_value)

        return "\n".join(collected).strip() if collected else None

    def call_specialist_agent(self, document_text):
        conversation = None

        try:
            conversation = self.openai_client.conversations.create()
            print(f"💬 Specialist conversation created: {conversation.id}")

            prompt = f"""Analyse cette carte d'identité française et renvoie UNIQUEMENT le JSON demandé.

TEXTE DU DOCUMENT:
{document_text}

Renvoie UNIQUEMENT du JSON valide, aucun autre texte.
"""

            response = self.openai_client.responses.create(
                conversation=conversation.id,
                input=prompt,
                extra_body={
                    "agent_reference": {
                        "name": self.specialist_agent_name,
                        "type": "agent_reference",
                    }
                },
            )

            result = self._extract_output_text(response)

            if result:
                print("✅ Specialist analysis completed")
                return result

            print("⚠️ Specialist agent returned no output_text")
            return json.dumps({
                "statut": "ERROR",
                "errore": "Specialist agent returned empty response"
            }, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Specialist analysis error: {e}")
            return json.dumps({
                "statut": "ERROR",
                "errore": f"Specialist agent failed: {str(e)}"
            }, ensure_ascii=False)

        finally:
            if conversation:
                try:
                    self.openai_client.conversations.delete(conversation_id=conversation.id)
                    print(f"🗑️ Specialist conversation deleted: {conversation.id}")
                except Exception as cleanup_error:
                    print(f"⚠️ Unable to delete specialist conversation: {cleanup_error}")

    def _submit_function_outputs_until_done(self, response):
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            tool_outputs = []

            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) != "function_call":
                    continue

                function_name = getattr(item, "name", None)
                if function_name != "analyze_french_id_document":
                    continue

                raw_arguments = getattr(item, "arguments", "{}") or "{}"

                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {}

                document_text = arguments.get("document_text", "")
                specialist_result = self.call_specialist_agent(document_text)

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": specialist_result,
                })

            if not tool_outputs:
                return response

            response = self.openai_client.responses.create(
                input=tool_outputs,
                previous_response_id=response.id,
                extra_body={
                    "agent_reference": {
                        "name": self.orchestrator_agent_name,
                        "type": "agent_reference",
                    }
                },
            )

        raise RuntimeError("Max function-call iterations reached before completion.")

    def analyze_document(self, file_path):
        document_text = self.extract_text_from_pdf(file_path)
        if not document_text:
            return json.dumps({
                "statut": "ERROR",
                "errore": "OCR failed",
                "api_type": "new_foundry_agents"
            }, ensure_ascii=False)

        conversation = None

        try:
            conversation = self.openai_client.conversations.create()
            print(f"💬 Orchestrator conversation created: {conversation.id}")

            collaborative_message = f"""J'ai besoin d'analyser une carte d'identité française. Voici le texte extrait du document:

{document_text}

Utilise obligatoirement la fonction analyze_french_id_document pour extraire les informations structurées depuis ce texte, puis retourne exactement le résultat de cette fonction.
"""

            response = self.openai_client.responses.create(
                conversation=conversation.id,
                input=collaborative_message,
                extra_body={
                    "agent_reference": {
                        "name": self.orchestrator_agent_name,
                        "type": "agent_reference",
                    }
                },
            )

            final_response = self._submit_function_outputs_until_done(response)
            final_text = self._extract_output_text(final_response)

            if final_text:
                print("✅ Orchestrator workflow completed")
                return final_text

            return json.dumps({
                "statut": "ERROR",
                "errore": "No response received from orchestrator",
                "api_type": "new_foundry_agents"
            }, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Orchestrator error: {e}")
            return json.dumps({
                "statut": "ERROR",
                "errore": f"Orchestrator failed: {str(e)}",
                "api_type": "new_foundry_agents"
            }, ensure_ascii=False)

        finally:
            if conversation:
                try:
                    self.openai_client.conversations.delete(conversation_id=conversation.id)
                    print(f"🗑️ Orchestrator conversation deleted: {conversation.id}")
                except Exception as cleanup_error:
                    print(f"⚠️ Unable to delete orchestrator conversation: {cleanup_error}")

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
            result["orchestrator_agent_name"] = self.orchestrator_agent_name
            result["specialist_agent_name"] = self.specialist_agent_name

            return result

        except Exception as e:
            return {
                "statut": "ERROR",
                "errore": f"JSON parse error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "api_type": "new_foundry_agents",
                "raw_response": response_text[:1000] if response_text else None,
                "orchestrator_agent_name": self.orchestrator_agent_name,
                "specialist_agent_name": self.specialist_agent_name,
            }

    def close(self):
        try:
            if self.project_client:
                self.project_client.close()
        except Exception:
            pass

        try:
            if self.credential:
                self.credential.close()
        except Exception:
            pass


def main():
    orchestrator = None

    try:
        pdf_path = os.getenv("PDF_PATH", "FacSimileID.pdf")

        print("🚀 Starting Document Analysis Orchestrator")
        print("📋 Mode: Azure AI Foundry New Foundry Agents")

        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return

        orchestrator = DocumentAnalysisOrchestrator()
        raw_result = orchestrator.analyze_document(pdf_path)

        print("\n📄 RESULT JSON (RAW):")
        print("=" * 60)
        print(raw_result)
        print("=" * 60)

        print("\n📋 EXTRACTED INFO:")
        print("=" * 60)

        parsed = orchestrator.parse_json_response(raw_result)

        if parsed.get("statut") == "ERROR":
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        else:
            print(f"📝 Cognome: {parsed.get('nom', 'N/A')}")
            print(f"📝 Nome/i: {parsed.get('prenoms', 'N/A')}")
            print(f"📅 Data di nascita: {parsed.get('date_naissance', 'N/A')}")
            print(f"📍 Luogo di nascita: {parsed.get('lieu_naissance', 'N/A')}")
            print(f"👤 Sesso: {parsed.get('sexe', 'N/A')}")
            print(f"🌍 Nazionalità: {parsed.get('nationalite', 'N/A')}")
            print(f"🆔 Numero carta: {parsed.get('numero_carte', 'N/A')}")
            if "confiance" in parsed:
                print(f"✅ Livello confidenza: {parsed.get('confiance', 'N/A')}")
            if "statut" in parsed:
                print(f"📊 Stato estrazione: {parsed.get('statut', 'N/A')}")

        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        if orchestrator:
            orchestrator.close()


if __name__ == "__main__":
    main()
