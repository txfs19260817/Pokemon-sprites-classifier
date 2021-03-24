FROM python:3.8-buster

WORKDIR /usr/src/app

# Install dependencies
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip3 install --no-cache-dir torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy necessary files
COPY utils ./utils
COPY app.py *.pem *.pth ./
RUN mkdir dataset
COPY dataset/label.csv dataset/label.csv
COPY configs ./configs

# Run
EXPOSE 14514
CMD [ "gunicorn", "app:app", "-c", "./configs/gunicorn.conf.py"]