import os
import json
from flask import Flask, request, jsonify, send_file
from google import genai
from google.genai import types

app = Flask(__name__)

# Core hardcoded key integration or secure pickup mapping fallback
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyA1dkLzmH57MeLQ8CzY5JrXwwZAIozHoW0")
client = genai.Client(api_key=API_KEY)

CHAT_HISTORY = []
AVAILABLE_SLOTS = ["Monday at 10:00 AM", "Monday at 2:00 PM", "Wednesday at 4:00 PM"]
BOOKINGS = []

# ==========================================
# DEFINING THE AGENT'S TOOLS
# ==========================================

def check_calendar() -> str:
    """Checks the internal business database for available open appointment time slots."""
    if not AVAILABLE_SLOTS:
        return json.dumps({"status": "empty", "available_slots": []})
    return json.dumps({"status": "success", "available_slots": AVAILABLE_SLOTS})

def create_booking(customer_name: str, email: str, selected_slot: str) -> str:
    """Creates a locked appointment booking inside the database for a customer."""
    if selected_slot not in AVAILABLE_SLOTS:
        return json.dumps({"success": False, "error": "That slot is no longer available."})
    
    AVAILABLE_SLOTS.remove(selected_slot)
    booking_id = f"BK-{len(BOOKINGS) + 1000}"
    BOOKINGS.append({
        "id": booking_id,
        "name": customer_name,
        "email": email,
        "slot": selected_slot
    })
    return json.dumps({"success": True, "booking_id": booking_id, "confirmed_slot": selected_slot})

def escalate_to_human(reason: str) -> str:
    """Triggers an emergency handoff to a live human employee if the customer is frustrated, angry, or confused."""
    print(f"\n🚨 [CRITICAL ESCALATION NOTICE]: {reason}\n")
    return json.dumps({"status": "escalated", "message": "A human support manager has been paged via internal alerts and is joining this interface right now."})

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
    return send_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global CHAT_HISTORY
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    CHAT_HISTORY.append({"role": "user", "parts": [{"text": user_message}]})
    
    config = types.GenerateContentConfig(
        system_instruction="""
            You are 'FrontDesk Elite AI', an upscale, polite customer concierge agent.
            
            CORE FEATURES & KNOWLEDGE MANDATES:
            1. STORE PROMOTIONS: We currently run a 'Summer Glow Makeover' promotion. It gives customers 20% off any store service if they book today. Enthusiastically pitch this deal if users ask about special offers, promos, or discounts.
            2. GENERAL STORE OPERATIONS: We are an elite local salon and wellness boutique offering premium styling, haircare, and facial treatments.
            3. CUSTOMER SUPPORT: Help users find bookings or answer common questions elegantly. 
            4. BOOKINGS: You must call 'check_calendar' to review slots. If the customer requests an appointment, collect their Name, Email, and preferred Slot, then call 'create_booking'.
            5. ESCALATIONS: If a human support connection is requested or negative user anger is detected, call 'escalate_to_human'.
        """,
        tools=[check_calendar, create_booking, escalate_to_human]
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=CHAT_HISTORY,
            config=config
        )
        
        if response.function_calls:
            for call in response.function_calls:
                tool_name = call.name
                arguments = call.args
                
                if tool_name in TOOL_MAP:
                    tool_output_str = TOOL_MAP[tool_name](**arguments)
                    tool_output = json.loads(tool_output_str)
                    
                    CHAT_HISTORY.append(response.candidates[0].content)
                    
                    CHAT_HISTORY.append({
                        "role": "tool",
                        "parts": [{
                            "function_response": {
                                "name": tool_name,
                                "response": {"result": tool_output_str}
                            }
                        }]
                    })
                    
                    # Intercept flow context explicitly to pass beautiful interactive cards for open calendar slots
                    if tool_name == "check_calendar" and tool_output.get("status") == "success":
                        return jsonify({
                            "ui_type": "slots",
                            "slots_data": tool_output.get("available_slots", [])
                        })
                    
                    final_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=CHAT_HISTORY,
                        config=config
                    )
                    CHAT_HISTORY.append(final_response.candidates[0].content)
                    return jsonify({"reply": final_response.text})
        
        CHAT_HISTORY.append(response.candidates[0].content)
        return jsonify({"reply": response.text})

    except Exception as e:
        print(f"CRASH ERROR LOGGED: {str(e)}")
        return jsonify({"reply": f"Backend Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)