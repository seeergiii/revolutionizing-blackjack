install_package:
	@pip install -e .

install_move_recommender:
	@pip install -e ./blackjack/move_recommender

install_computer_vision:
	@pip install -e ./blackjack/computer_vision

install_frontend:
	@pip install -e ./blackjack/frontend

run_api:
	uvicorn blackjack.computer_vision.api:app --reload
