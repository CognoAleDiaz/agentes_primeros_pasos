"""
Daily Special Days Email Agent
================================
A LangGraph agent that searches for fun/special days (Spain + international),
picks the best 1-2, and drafts an email about them.

Graph flow: search_days → select_days → draft_email
"""

import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

# Load environment variables (TAVILY_API_KEY lives in .env)
load_dotenv()

# ---------------------------------------------------------------------------
# Azure OpenAI connection (Cognodata)
# Reads credentials from .env file — keeps secrets out of the code
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    base_url=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    model=os.getenv("AZURE_DEPLOYMENT"),
    temperature=0.7,  # A bit of creativity for fun email drafts
)


# ---------------------------------------------------------------------------
# State definition — this is the data that flows through the graph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    date: str               # Today's date formatted as "16 de abril de 2026"
    search_results: str     # Raw search results from Tavily
    selected_days: str      # The 1-2 best days chosen by the LLM
    email_draft: str        # Email draft ready to send
    email_draft_checked: str # Email draft checked for accuracy and grammar


# ---------------------------------------------------------------------------
# Node 1: Search for special days using Tavily web search
# We run two queries — one for Spain-specific days, one for international days
# ---------------------------------------------------------------------------
def search_days(state: AgentState) -> dict:
    """Search the web for special/fun days happening today."""

    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    date = state["date"]

    # Two queries to cover both Spanish and international celebrations
    query_spain = f"qué día especial se celebra hoy {date} en España"
    query_international = f"what international day is celebrated today {date}"

    # Tavily search returns a list of results with content snippets
    results_spain = tavily.search(query=query_spain, max_results=5)
    results_international = tavily.search(query=query_international, max_results=5)

    # Combine all result snippets into a single text block for the LLM
    all_results = []
    for r in results_spain["results"]:
        all_results.append(f"[España] {r['content']}")
    for r in results_international["results"]:
        all_results.append(f"[Internacional] {r['content']}")

    combined = "\n\n".join(all_results)

    return {"search_results": combined}


# ---------------------------------------------------------------------------
# Node 2: LLM picks the 1-2 most fun and special days from the search results
# ---------------------------------------------------------------------------
def select_days(state: AgentState) -> dict:
    """Use the LLM to choose the most fun and interesting days from the search results."""

    prompt = f"""Hoy es {state['date']}. A continuación tienes información sobre días especiales
        que se celebran hoy, tanto en España como internacionalmente:

        {state['search_results']}

        Tu tarea:
                ### FASE 1: FILTRADO (CRÍTICO)
                Antes de elegir, analiza la lista de días disponibles y elimina inmediatamente cualquier día que cumpla con estos criterios:
                1. Conmemoraciones de tragedias, guerras, genocidios o desastres.
                2. Temas políticos, religiosos o conflictos sociales.
                3. Días de concienciación médica grave (enfermedades, luto).

                ### FASE 2: SELECCIÓN
                De la lista restante (solo eventos positivos, curiosos o festivos), selecciona los 2 días que tengan mayor potencial para:
                - Tienen que ser los mas famosos o relevantes para una consultora tecnologica.
                - Es decir, tiene que ser un dia muy muy reconocido, o algo que sea importante para la empresa
                - Un poco mas de contexto sobre la empresa: 
                - Cognodata es una consultora estratégica especializada en Data Science, Inteligencia Artificial y Big Data 
                        que ayuda a las empresas a transformar su información en decisiones de negocio de alto impacto. 
                        Su enfoque combina una sólida implementación tecnológica con metodologías avanzadas 
                        para optimizar procesos y personalizar la experiencia del cliente.
                - Generar una sonrisa.
                - Iniciar una conversación ligera.
                - Ser tendencia en redes sociales.
                - Solo incluir dias sobre comida si es algo muy muy especial para los españoles.

            Responde SOLO con los días elegidos y sus descripciones, nada más. TODO EN ESPAÑOL (MUY IMPORTANTE)
    """

    response = llm.invoke(prompt)
    return {"selected_days": response.content}


# ---------------------------------------------------------------------------
# Node 3: Draft the email based on the selected days
# This is a placeholder system prompt — the user will refine it later
# ---------------------------------------------------------------------------
def draft_email(state: AgentState) -> dict:
    """Draft a fun, engaging email about today's special day(s)."""

    system_prompt = """Eres un redactor de emails divertido y creativo para una empresa.
            Tu trabajo es escribir un email corto y alegre sobre el día especial de hoy.

            Reglas:
            - El email debe ser informal pero profesional
            - Incluye datos curiosos y relevantes sobre el día
            - Máximo 150 palabras
            - Incluye un asunto llamativo para el email
            - Incluye emoticonos 
            - No incluyas palabras en ingles, a menos que sea absolutamente necesario.
            - El cuerpo del email debe estar en formato HTML (usa <b> para negritas, <br> para saltos de línea, etc.)
            - NO uses markdown (como **texto**). Usa etiquetas HTML.
            - Formato: ASUNTO: (el asunto del email)   CUERPO:(el cuerpo del email) Un saludo, [Tu nombre]

            EJEMPLO DE EMAIL:
                📚 Hoy celebramos el Día del Libro 📚

                    Cada 23 de abril se celebra el Día Mundial del Libro y del Derecho de Autor (y el cumpleaños de Fernando corbacho y Claudia Ceica 😊), una fecha impulsada por la UNESCO para rendir homenaje a los libros, a la lectura y a quienes los escriben

                📖 ¿Qué es el Día del Libro?

                    Es un día para poner en valor el poder de los libros como fuente de conocimiento, cultura, imaginación y aprendizaje. La UNESCO lo instauró oficialmente en 1995 con el objetivo de fomentar la lectura y proteger la creatividad de autoras y autores

                📅 ¿Por qué se celebra el 23 de abril?

                        Porque en esta fecha, en 1616, fallecieron grandes figuras de la literatura universal como Miguel de Cervantes, William Shakespeare y el Inca Garcilaso de la Vega, lo que convierte este día en un símbolo para la literatura a nivel mundial

                🌍 ¿Qué se hace en este día?

                    En muchos países se organizan:

                    Ferias del libro y firmas de ejemplares
                    Lecturas públicas y encuentros con autoras y autores
                    Actividades en bibliotecas, colegios y empresas
                    En algunas zonas, como Cataluña, es tradición regalar un libro y una rosa, uniendo cultura y cariño 🌹📘

                    En Madrid han puesto una “biblioteca” al aire libre: está en el matadero.

                🏢 ¿Qué podemos hacer en la oficina?

                    Aprovechemos el día para compartir nuestra pasión lectora:

                    📚 Recomendar ese libro que nos marcó (o el que tenemos en la mesilla)
                    💬 Compartir citas literarias que nos inspiran
                    🔄 Intercambiar libros entre compañer@s
                    ☕ Reservar un ratito para leer, aunque sea unas páginas
                    Porque los libros abren la mente, generan conversación y nos conectan… incluso en el trabajo 😊

                ✨ Feliz Día del Libro ✨

                ¿Qué libro recomendarías hoy? Yo os dejo un Excel compartido con los libros que he leído este año, podéis añadir más 
    """

    prompt = f"""{system_prompt}

            Hoy es {state['date']}. Los días especiales de hoy son:

            {state['selected_days']}

            Escribe el email.
    """

    response = llm.invoke(prompt)

    return {"email_draft": response.content}


# ---------------------------------------------------------------------------
# Node 3: Check the email draft for accuracy and grammar
# ---------------------------------------------------------------------------
def check_email_draft(state: AgentState) -> dict:
    """Check the email draft for accuracy and grammar."""

    system_prompt = """Eres un corrector de emails.
            Tu trabajo es revisar el email draft y corregir cualquier error de gramática o ortografía.
            Asegúrate de que todo tiene sentido, se lee de forma fluida y es coherente.
            Asegurate tambien de que no haya palabras en ingles, a menos que sea absolutamente necesario.
            El formato es primero el asunto y luego el cuerpo del email.
            Asegurate de que el email termina en Un saludo,
            Asegurate de que no hay nada repetido en el email.
            IMPORTANTE: El cuerpo del email está en formato HTML. Mantén todas las etiquetas HTML (<b>, <br>, <p>, etc.) intactas.
            Asegúrate de que hay etiquetas <br> para los saltos de línea entre párrafos y secciones.
            Asegúrate de que hay etiquetas <br> justo después del saludo inicial,
            Asegúrate de que hay etiquetas <br> justo antes del Un saludo,
            Asegúrate de que hay una etiqueta <br> justo despues del Un saludo, y antes del nombre del remitente.
            POR ULTIMO: ASEGURATE DE QUE LOS <br> son dobles, es decir convierte <br> en <br><br>
            Solo responde con el email corregido, nada más.
    """

    prompt = f"""{system_prompt}
            El email draft es:
            {state['email_draft']}
    """

    response = llm.invoke(prompt)

    return {"email_draft_checked": response.content}


# ---------------------------------------------------------------------------
# Node 4: Send the email
# ---------------------------------------------------------------------------
def send_email(state: AgentState) -> dict:
    """Send the email via SMTP (Office 365) to all recipients in USER_EMAILS."""
    
    # Parse the checked email draft — expects "ASUNTO: ...\n\nCUERPO..."
    draft = state["email_draft_checked"]
    draft = draft.replace("*", "")
    lines = draft.strip().split("\n", 1)

    for i, line in enumerate(lines):
        print(f"Line {i}: {line}")    


    # Extract subject from the first line, rest is the body
    subject = lines[0].replace("ASUNTO:", "").replace("Asunto:", "").strip()
    body = lines[1].strip() if len(lines) > 1 else draft
    body = body.replace("CUERPO: ", "")
    body = body.replace("[Tu nombre]", os.getenv("USER_NAME"))

    # SMTP credentials from .env
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.office365.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    # List of recipients from .env (comma-separated)
    recipients = [e.strip() for e in os.getenv("USER_EMAILS").split(",")]

    # Build and send the email via SMTP with TLS
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html", "utf-8"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, msg.as_string())

    print("=" * 60)
    print(f"✅ Email enviado a: {', '.join(recipients)}")
    print("=" * 60)
    
    return {"email_sent": True}
    
    

# ---------------------------------------------------------------------------
# Build the LangGraph — linear flow: search → select → draft → revised_draft → send → END
# ---------------------------------------------------------------------------
def build_graph():
    """Assemble and compile the 3-node LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add the three processing nodes
    graph.add_node("search_days", search_days)
    graph.add_node("select_days", select_days)
    graph.add_node("draft_email", draft_email)
    graph.add_node("check_email_draft", check_email_draft)
    graph.add_node("send_email", send_email)

    # Wire them in sequence: START → search → select → draft → revised_draft → END
    graph.add_edge(START, "search_days")
    graph.add_edge("search_days", "select_days")
    graph.add_edge("select_days", "draft_email")
    graph.add_edge("draft_email", "check_email_draft")
    graph.add_edge("check_email_draft", "send_email")
    graph.add_edge("send_email", END)

    return graph.compile()
