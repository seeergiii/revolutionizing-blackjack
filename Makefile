##### Run API - - - - - - - - - - - - - - - - - - - - - - - - -
run_api:
	uvicorn revolutionizing-blackjack.backend.fast_api.api:app --reload

##### Run Interface (Streamlit) - - - - - - - - - - - - - - - - - - - - - - - - -
run_interface:
	streamlit run frontend/app.py
