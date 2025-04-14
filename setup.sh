#!/usr/bin/env bash

# Создание нового виртуального окружения
python3.11 -m venv hellhole
source hellhole/bin/activate

echo "You have entered the hole"

# Обновим pip, just in case
pip install --upgrade pip

# Устанавливаем PyTorch с поддержкой MPS (для чипов M)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Остальные зависимости
pip install tqdm==4.66.1
pip install requests==2.31.0
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install scikit-learn==1.2.2
pip install numpy==1.26.3
pip install tokenizers==0.13.3
pip install sentencepiece==0.1.99

# Скачиваем модель
#curl -O https://www.cs.cmu.edu/~vijayv/stories42M.pt


#curl -O https://www.cs.cmu.edu/~vijayv/stories42M.pt
