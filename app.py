import os
import json
from flask import Flask, request, jsonify, send_file
from google import genai
from google.genai import types

app = Flask(__name__)

# Initialize the Gemini Client (Picks up GEMINI_API_KEY from environment variables)
client = genai.Client()

# Core Session Memory (In production, map this to a database using a unique session ID)
CHAT_HISTORY = []

# Mock internal database state
AVAILABLE_SLOTS = ["Monday at 10:00 AM", "Monday at 2:00 PM", "Wednesday at 4:00 PM"]
BOOKINGS = []

# ==========================================
# DEFINING THE AGENT'S PYTHON TOOLS
# ==========================================

def check_calendar() -> str:
    """Checks the internal business database for available open appointment time slots."""
    if not AVAILABLE_SLOTS:
        return "No open slots available at the moment."
    return json.dumps({"available_slots": AVAILABLE_SLOTS})

def create_booking(customer_name: str, email: str, selected_slot: str) -> str:
    """Creates a locked appointment booking inside the database for a customer."""
    if selected_slot not in AVAILABLE_SLOTS:
        return json.dumps({"success": False, "error": "That slot is no longer available."})
    
    # Process the booking action
    AVAILABLE_SLOTS.remove(selected_slot)
    booking_id = f"BK-{len(BOOKINGS) + 1000}"
    new_booking = {
        "id": booking_id,
        "name": customer_name,
        "email": email,
        "slot": selected_slot
    }
    BOOKINGS.append(new_booking)
    return json.dumps({"success": True, "booking_id": booking_id, "confirmed_slot": selected_slot})

def escalate_to_human(reason: str) -> str:
    """Triggers an emergency handoff to a live human employee if the customer is frustrated, angry, or confused."""
    print(f"\n🚨 [CRITICAL ESCALATION NOTICE]: {reason}\n")
    return json.dumps({"status": "escalated", "message": "A manager has been paged and is jumping into this chat thread immediately."})

# Mapping strings to actual executable references
TOOL_MAP = {
    "check_calendar": check_calendar,
    "create_booking": create_booking,
    "escalate_to_human": escalate_to_human
}

# ==========================================
# FLASK ROUTING LOGIC
# ==========================================

@app.route('/')
def home():
    # Serves the front-end chat interface directly
    return send_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global CHAT_HISTORY
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Append user input into the global thread state
    CHAT_HISTORY.append(types.Content(role="user", parts=[types.Part.from_text(user_message)]))
    
    # Configure system guardrails and hand over Python functions as tools
    config = types.GenerateContentConfig(
        system_instruction="""
            You are 'FrontDesk AI', a highly competent, friendly, and practical assistant for a local small business.
            Your job is to answer queries, check calendar availability, and lock in direct bookings.
            
            OPERATIONAL MANDATES:
            1. If a user wants to book, you MUST first execute 'check_calendar' to see what's actually free. 
            2. Never guess or make up appointment slots.
            3. To finalize a booking, you must collect their full name, email, and explicit time preference, then call 'create_booking'.
            4. SAFETY RULE: If the customer uses foul language, shows strong signs of anger, or demands human intervention, immediately invoke the 'escalate_to_human' tool.
        """,
        tools=[check_calendar, create_booking, escalate_to_human]
    )
    
    # Run the core Observe-Think-Act reasoning loop
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=CHAT_HISTORY,
            config=config
        )
        
        # Check if the model decided to execute an action (Function Call)
        if response.function_calls:
            for call in response.function_calls:
                tool_name = call.name
                arguments = call.args
                
                # Execute the mapped python tool matching the model's call request
                if tool_name in TOOL_MAP:
                    tool_output = TOOL_MAP[tool_name](**arguments)
                    
                    # Log the historical track of the tool invocation
                    CHAT_HISTORY.append(response.candidates[0].content)
                    
                    # Provide the function output response back into the context
                    CHAT_HISTORY.append(types.Content(
                        role="user", 
                        parts=[types.Part.from_function_response(name=tool_name, response={"result": tool_output})]
                    ))
                    
                    # Second model pass allows the AI to translate the tool data back into plain language
                    final_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=CHAT_HISTORY,
                        config=config
                    )
                    CHAT_HISTORY.append(final_response.candidates[0].content)
                    return jsonify({"reply": final_response.text})
        
        # Standard textual response path
        CHAT_HISTORY.append(response.candidates[0].content)
        return jsonify({"reply": response.text})

    except Exception as e:
        return jsonify({"reply": f"System engine error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)