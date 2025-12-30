"""
LangGraph Agent for CogniAnchor
Intelligent agent with tool-calling capabilities for dementia care
"""

import logging
import os
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.agent_tools import (
    create_reminder,
    list_reminders,
    delete_reminder,
    send_emergency_alert
)

logger = logging.getLogger("LangGraphAgent")

# === Agent State ===

class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[Sequence[BaseMessage], "The conversation messages"]
    pair_id: str
    patient_id: str


# === System Prompt for Agent ===

AGENT_SYSTEM_PROMPT = """You are a compassionate AI companion for patients with cognitive challenges (dementia/Alzheimer's).

Your role:
- Provide warm, patient, and clear communication
- Help manage daily tasks through reminders
- Offer emotional support and reassurance
- Monitor for signs of distress or danger
- Use simple, short sentences (maximum 2 sentences per response)

Available Tools:
1. create_reminder: Set reminders for medication, appointments, tasks
2. list_reminders: Show upcoming reminders
3. delete_reminder: Cancel/remove reminders
4. send_emergency_alert: Alert caregiver if patient is in danger (USE SPARINGLY - only for real emergencies)

Guidelines:
- Always use tools when the patient asks for reminder-related actions
- Be proactive: If patient mentions taking medicine or appointments, suggest creating a reminder
- Never show frustration or correct the patient harshly
- Validate their feelings and provide reassurance
- For emergency situations (fall, pain, severe confusion), use send_emergency_alert tool
- Keep responses brief and warm

Examples:
- Patient: "Remind me to take my pills at 8pm" -> Use create_reminder tool
- Patient: "What do I need to do today?" -> Use list_reminders tool
- Patient: "Cancel my appointment reminder" -> Use delete_reminder tool
- Patient: "I fell and I can't get up" -> Use send_emergency_alert tool immediately
- Patient: "I'm feeling sad" -> Provide emotional support (no tool needed)
"""


# === Initialize LLM with Tools ===

def create_agent_llm():
    """Create LLM with tool binding"""
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        # Initialize Gemini with tool support (using Gemini 1.5 Pro for better quota)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=gemini_api_key,
            temperature=0.7,
            max_output_tokens=500
        )

        # Bind tools to the LLM
        tools = [create_reminder, list_reminders, delete_reminder, send_emergency_alert]
        llm_with_tools = llm.bind_tools(tools)

        logger.info("Agent LLM initialized successfully with Gemini and tools")
        return llm_with_tools, tools

    except Exception as e:
        logger.error(f"Failed to initialize agent LLM: {e}")
        raise


# === Agent Nodes ===

def call_agent(state: AgentState):
    """
    Agent node: Calls the LLM to decide next action
    """
    try:
        llm_with_tools, _ = create_agent_llm()

        # Build messages with system prompt
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        ] + state["messages"]

        # Call LLM
        response = llm_with_tools.invoke(messages)

        logger.info(f"Agent response: {response.content if hasattr(response, 'content') else 'Tool call'}")

        # Return updated state
        return {"messages": [response]}

    except Exception as e:
        logger.error(f"Error in agent node: {e}")
        # Return error message
        error_msg = AIMessage(content="I'm having trouble right now. Let me try to help you anyway.")
        return {"messages": [error_msg]}


def should_continue(state: AgentState):
    """
    Conditional edge: Determine if we should continue to tools or end
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"Agent calling {len(last_message.tool_calls)} tool(s)")
        return "tools"

    # Otherwise, end the conversation turn
    logger.info("Agent finished - no more tool calls")
    return END


# === Build Agent Graph ===

def create_agent_graph():
    """
    Create the LangGraph agent workflow

    Graph structure:
    START -> agent -> [should_continue] -> tools -> agent -> END
                                        -> END
    """
    try:
        # Get tools for ToolNode
        _, tools = create_agent_llm()

        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", call_agent)
        workflow.add_node("tools", ToolNode(tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")

        # Compile the graph
        app = workflow.compile()

        logger.info("Agent graph created successfully")
        return app

    except Exception as e:
        logger.error(f"Failed to create agent graph: {e}")
        raise


# === Agent Execution ===

async def run_agent(patient_id: str, pair_id: str, message: str, conversation_history: list = None) -> str:
    """
    Run the agent with a user message

    Args:
        patient_id: Patient identifier
        pair_id: Patient-caretaker pair ID
        message: User's message
        conversation_history: Previous conversation messages (optional)

    Returns:
        Agent's response text
    """
    try:
        logger.info(f"Running agent for patient {patient_id}: {message}")

        # Create agent graph
        agent = create_agent_graph()

        # Build initial state
        messages = []

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=message))

        initial_state = {
            "messages": messages,
            "pair_id": pair_id,
            "patient_id": patient_id
        }

        # Run the agent
        final_state = agent.invoke(initial_state)

        # Extract final response
        final_messages = final_state["messages"]

        # Get the last AI message
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content:
                logger.info(f"Agent completed successfully")
                return msg.content

        # Fallback if no AI message found
        logger.warning("No AI response found in final state")
        return "I'm here to help. What would you like to know?"

    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return "I'm having some trouble right now, but I'm here with you. How can I help?"


# === Conversation Memory (In-Memory for now) ===

# In production, store this in database
agent_conversations = {}

def get_agent_history(patient_id: str):
    """Get conversation history for a patient"""
    if patient_id not in agent_conversations:
        agent_conversations[patient_id] = []
    return agent_conversations[patient_id]

def add_to_agent_history(patient_id: str, role: str, content: str):
    """Add message to conversation history"""
    if patient_id not in agent_conversations:
        agent_conversations[patient_id] = []

    agent_conversations[patient_id].append({
        "role": role,
        "content": content
    })

    # Keep only last 10 messages
    if len(agent_conversations[patient_id]) > 10:
        agent_conversations[patient_id] = agent_conversations[patient_id][-10:]
