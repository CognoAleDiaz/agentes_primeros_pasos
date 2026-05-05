"""
Daily Special Days Email Agent
================================
A LangGraph agent that searches for fun/special days (Spain + international),
picks the best 1-2, and drafts an email about them.

Graph flow: search_days → select_days → draft_email
"""

import os
import re
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict

# Maximum number of select → evaluate retries before giving up entirely
MAX_RELEVANCE_ATTEMPTS = 3

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
    selected_days: str      # The 1-2 best days chosen by the LLM (narrowed to only relevant ones after evaluation)
    relevance_decision: str # "SI" or "NO" — whether at least one selected day is relevant
    relevance_reason: str   # LLM's explanation for the relevance decision
    relevance_attempts: int # How many select → evaluate cycles have run (max MAX_RELEVANCE_ATTEMPTS)
    rejected_days: str      # Accumulated text of days rejected in previous attempts, fed back to select_days
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

    rejected = (state.get("rejected_days") or "").strip()
    rejected_clause = ""
    if rejected:
        rejected_clause = f"""
        ATENCIÓN — REINTENTO:
        En intentos anteriores ya se descartaron los siguientes días por NO ser
        suficientemente relevantes. NO los vuelvas a seleccionar. Elige días DISTINTOS:

        {rejected}
        """

    prompt = f"""Hoy es {state['date']}. A continuación tienes información sobre días especiales
        que se celebran hoy, tanto en España como internacionalmente:

        {state['search_results']}
        {rejected_clause}
        Tu tarea:
                FASE 1: FILTRADO (CRÍTICO)
                Antes de elegir, analiza la lista de días disponibles y elimina inmediatamente cualquier día que cumpla con estos criterios:
                1. Conmemoraciones de tragedias, guerras, genocidios o desastres.
                2. Temas políticos, religiosos o conflictos sociales.
                3. Días de concienciación médica grave (enfermedades, luto).

                FASE 2: SELECCIÓN
                De la lista restante (solo eventos positivos, curiosos o festivos), selecciona 1 o 2 días (máximo 2) que tengan mayor potencial para:
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

                IMPORTANTE: Si tras el filtrado NO queda ningún día que cumpla los criterios (o solo quedan días irrelevantes, tristes, polémicos o sin gancho),
                responde EXACTAMENTE con la palabra: NINGUNO
                No inventes días ni fuerces la selección. Es preferible no enviar correo a enviar uno sobre un día irrelevante.

                Si encuentras solo 1 día relevante, responde solo con ese. Si encuentras 2, responde con los 2.

            Responde SOLO con los días elegidos y sus descripciones, nada más (o "NINGUNO" si no hay). TODO EN ESPAÑOL (MUY IMPORTANTE)
    """

    response = llm.invoke(prompt)
    return {"selected_days": response.content}


# ---------------------------------------------------------------------------
# Node 2.5: LLM evaluates each preselected day individually
# - If both are relevant → keep both, decision = SI
# - If only one is relevant → narrow selected_days to just that one, decision = SI
# - If none are relevant → decision = NO, increment attempt counter,
#   record the rejected days so the next select_days call avoids them
# ---------------------------------------------------------------------------
def evaluate_relevance(state: AgentState) -> dict:
    """Evaluate each selected day individually and either narrow the selection or trigger a retry."""

    selected = (state.get("selected_days") or "").strip()
    attempts = state.get("relevance_attempts", 0) or 0
    prev_rejected = state.get("rejected_days") or ""

    # Shortcut: if select_days already returned NINGUNO, count it as a failed attempt
    if not selected or selected.upper().startswith("NINGUNO"):
        new_attempts = attempts + 1
        return {
            "relevance_decision": "NO",
            "relevance_reason": "El nodo de selección no encontró ningún día candidato (respondió NINGUNO).",
            "relevance_attempts": new_attempts,
            "rejected_days": prev_rejected + f"\n[Intento {new_attempts}]: (sin candidatos)",
        }

    prompt = f"""Hoy es {state['date']}. Eres un evaluador EXIGENTE.
        Tu trabajo es decidir si los siguientes días preseleccionados merecen un email a los empleados de Cognodata:

        {selected}

        CONTEXTO DE LA EMPRESA (CRÍTICO):
        Cognodata es una consultora estratégica especializada en Data Science, Inteligencia Artificial y Big Data.
        Sus empleados son perfiles técnicos: data scientists, ingenieros de datos, consultores de IA, analistas.
        Los emails son internos para el equipo. Cognodata tiene operaciones tanto en España como en Mexico.

        FILOSOFÍA DE EVALUACIÓN:
        - Por defecto, la respuesta es NO. La mayoría de los "días internacionales de X" son irrelevantes y no merecen un email.
        - Es MUCHO mejor no enviar correo que enviar uno mediocre, forzado o aburrido.
        - Solo aprueba un día si genuinamente justificarías mandarlo a 50 profesionales ocupados.
        - "Está bien", "es simpático", "podría interesar a alguien" → NO es suficiente. RECHAZA.

        UN DÍA SOLO ES RELEVANTE SI CUMPLE AL MENOS UNO DE ESTOS CRITERIOS DUROS:
        1. Está directamente relacionado con tecnología, ciencia, datos, IA, internet, programación o innovación
           (ej: Día de Internet, Día del Programador, Día de la Ciencia, Día del Pi).
        2. Es una celebración cultural española o mexicana MUY conocida y con tradición real
           (ej: San Juan, Día del Libro 23 abril, Día de la Hispanidad, Día de la Madre/Padre).
        3. Es un evento global universalmente reconocido que cualquier adulto identifica al instante
           (ej: Halloween, San Valentín, Año Nuevo Chino, Día de la Mujer 8M).

        RECHAZA AUTOMÁTICAMENTE (sin excepciones):
        - Días de comida o bebida específica salvo que sea ICÓNICO para España (ej: rechaza "Día del Donut", "Día del Aguacate").
        - Días de animales, plantas, hobbies de nicho o curiosidades raras.
        - Días promocionales/comerciales inventados por marcas o industrias.
        - Días de concienciación sobre enfermedades, problemas sociales, tragedias, conflictos.
        - Días religiosos, políticos, o ideológicamente sensibles.
        - Cualquier día que necesites buscar en Google para entender qué es.
        - Cualquier día que un data scientist ocupado consideraría spam.

        TAREA:
        Evalúa CADA día por separado contra los criterios de arriba. Sé estricto y RAZONA EN VOZ ALTA.
        Para cada día tienes que listar pros y contras concretos, y emitir un veredicto.
        - Si AMBOS pasan el listón alto → mantén los dos.
        - Si SOLO UNO pasa → quédate solo con ese.
        - Si NINGUNO pasa (caso más probable) → NINGUNO.

        Responde EXACTAMENTE en este formato y nada más:

        ANALISIS:
        Día 1: <nombre del día>
        Pros:
        - <pro concreto>
        - <otro pro si aplica>
        Contras:
        - <contra concreto>
        - <otro contra si aplica>
        Veredicto: INCLUIR  o  DESCARTAR  (justificado en una frase mencionando el criterio específico)

        Día 2: <nombre del día>   (omite este bloque si solo había un día preseleccionado)
        Pros:
        - <pro concreto>
        Contras:
        - <contra concreto>
        Veredicto: INCLUIR  o  DESCARTAR  (justificado en una frase)

        DECISION: SI    (si al menos un Veredicto fue INCLUIR)  o  NO    (si todos fueron DESCARTAR)
        RAZON: <una frase de síntesis que resuma la decisión global>
        DIAS_RELEVANTES:
        <Copia aquí, palabra por palabra, SOLO los días con veredicto INCLUIR, con sus descripciones tal y como aparecen arriba. Si la decisión es NO, escribe únicamente NINGUNO.>

        TODO EN ESPAÑOL.
    """

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Parse the four sections.
    # ANALISIS is multi-line and runs from "ANALISIS:" up to "DECISION:".
    # RAZON is a single-line synthesis between DECISION and DIAS_RELEVANTES.
    # DIAS_RELEVANTES is multi-line and runs to end of text.
    analysis_match = re.search(
        r"AN[AÁ]LISIS:\s*(.*?)(?=^\s*DECISION:|\Z)",
        content,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    decision_match = re.search(r"^\s*DECISION:\s*(.+)$", content, re.IGNORECASE | re.MULTILINE)
    reason_match = re.search(
        r"RAZ[OÓ]N:\s*(.*?)(?=^\s*DIAS_RELEVANTES:|\Z)",
        content,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    days_match = re.search(r"DIAS_RELEVANTES:\s*(.*)", content, re.IGNORECASE | re.DOTALL)

    analysis = analysis_match.group(1).strip() if analysis_match else ""
    decision_raw = decision_match.group(1).strip().upper() if decision_match else "NO"
    decision = "SI" if decision_raw.startswith("SI") or decision_raw.startswith("SÍ") else "NO"
    short_reason = reason_match.group(1).strip() if reason_match else ""
    relevant_days = days_match.group(1).strip() if days_match else ""

    # Build the detailed reason: pros/cons analysis followed by the one-line synthesis.
    # If the model didn't follow the format, fall back to the raw content so we never lose its reasoning.
    if analysis or short_reason:
        parts = []
        if analysis:
            parts.append("ANÁLISIS POR DÍA:\n" + analysis)
        if short_reason:
            parts.append("SÍNTESIS: " + short_reason)
        reason = "\n\n".join(parts)
    else:
        reason = content

    # Treat an empty or NINGUNO body as a NO regardless of what the model claimed
    if not relevant_days or relevant_days.upper().startswith("NINGUNO"):
        decision = "NO"

    if decision == "SI":
        return {
            "relevance_decision": "SI",
            "relevance_reason": reason,
            "selected_days": relevant_days,  # narrowed to only the relevant subset
            "relevance_attempts": attempts,
        }

    new_attempts = attempts + 1
    return {
        "relevance_decision": "NO",
        "relevance_reason": reason,
        "relevance_attempts": new_attempts,
        "rejected_days": prev_rejected + f"\n[Intento {new_attempts}]: {selected}",
    }


# ---------------------------------------------------------------------------
# Conditional router:
#   - SI                     → draft_email
#   - NO + attempts < MAX    → retry (loop back to select_days with a new pick)
#   - NO + attempts >= MAX   → skip (give up, no email today)
# ---------------------------------------------------------------------------
def has_relevant_days(state: AgentState) -> str:
    """Route based on the LLM's relevance decision and how many attempts have been made."""
    decision = (state.get("relevance_decision") or "").strip().upper()
    attempts = state.get("relevance_attempts", 0) or 0
    reason = state.get("relevance_reason") or "(sin razón proporcionada)"

    if decision == "SI":
        print("=" * 60)
        print("✅ Días relevantes encontrados. Procediendo a redactar el email.")
        print("-" * 60)
        print(reason)
        print("=" * 60)
        return "draft_email"

    if attempts >= MAX_RELEVANCE_ATTEMPTS:
        print("=" * 60)
        print(f"⏭️  No se envía correo hoy (tras {attempts} intentos sin días relevantes).")
        print("-" * 60)
        print(reason)
        print("=" * 60)
        return "skip"

    print("=" * 60)
    print(f"🔄 Intento {attempts}/{MAX_RELEVANCE_ATTEMPTS}: ningún día relevante.")
    print("-" * 60)
    print(reason)
    print("   Reintentando con días distintos...")
    print("=" * 60)
    return "retry"


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
            - Formato: ASUNTO: (el asunto del email)   CUERPO: (el cuerpo del email) Un saludo, [Tu nombre]

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
            Tu trabajo es revisar el email y corregir cualquier error.
            - Asegúrate de que todo tiene sentido, se lee de forma fluida y es coherente.
            - Asegurate tambien de que no haya palabras en ingles, a menos que sea absolutamente necesario.
            - El formato es primero el asunto y luego el cuerpo del email.
            - Asegurate de que el email termina en Un saludo,
            - Asegurate de que no hay nada repetido en el email.
            - IMPORTANTE: El cuerpo del email está en formato HTML. Mantén todas las etiquetas HTML (<b>, <br>, <p>, etc.) intactas.
            - Asegúrate de que hay etiquetas <br> para los saltos de línea entre párrafos y secciones.
            - Asegúrate de que hay etiquetas <br> justo después del saludo inicial,
            - Asegúrate de que hay etiquetas <br> justo antes del Un saludo,
            - Asegúrate de que hay una etiqueta SOLO UN <br> justo despues del Un saludo, y antes del nombre del remitente.
            - POR ULTIMO: ASEGURATE DE QUE LOS <br> son dobles, es decir convierte <br> en <br><br> a menos que se indique lo contrario.
            - Solo responde con el email corregido, nada más.
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

    # Add the processing nodes
    graph.add_node("search_days", search_days)
    graph.add_node("select_days", select_days)
    graph.add_node("evaluate_relevance", evaluate_relevance)
    graph.add_node("draft_email", draft_email)
    graph.add_node("check_email_draft", check_email_draft)
    graph.add_node("send_email", send_email)

    # Wire them in sequence, with a conditional skip if the LLM judges the days irrelevant
    graph.add_edge(START, "search_days")
    graph.add_edge("search_days", "select_days")
    graph.add_edge("select_days", "evaluate_relevance")
    graph.add_conditional_edges(
        "evaluate_relevance",
        has_relevant_days,
        {"draft_email": "draft_email", "retry": "select_days", "skip": END},
    )
    graph.add_edge("draft_email", "check_email_draft")
    graph.add_edge("check_email_draft", "send_email")
    graph.add_edge("send_email", END)

    return graph.compile()
