prepare_local:
	pip install --upgrade -q git+https://github.com/keras-team/keras-cv
	pip install -e .
	mkdir ~/.data/blackjack
	mkdir ~/.data/blackjack/weights_checkpoints
	mkdir ~/.data/blackjack/weights_final
	mkdir ~/.data/blackjack/weights_start
	mkdir ~/.data/blackjack/data

prepare_cloud_folders:
		mkdir .data
		mkdir .data/weights_checkpoints
		mkdir .data/weights_final
		mkdir .data/weights_start

		mkdir .data/data
		unzip -u "data.zip" -d ".data/data"
