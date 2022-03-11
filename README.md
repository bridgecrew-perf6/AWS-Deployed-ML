<h1>AWS-Deployed-ML</h1> 
<img src="https://user-images.githubusercontent.com/45767042/157892991-811c3418-ed4e-4936-a820-ca73a00c04b2.png">

What we are familiar with, especially from the Kaggle platform, to pass the training and test data we have through the preprocessing steps on the notebook and submit the final estimate to the competition as submission. However, training every time the machine learning model to make prediction is not sufficient. The part I will talk about in this project is to train a machine learning model and deploy it on the AWS EC2 service in a way that is accessible to everyone and can make instant predictions.
The dataset and machine learning model I will use are the Titanic dataset and the CatBoost machine learning model, which I have previously analyzed and shared.

If you want to review those analyses:
* As Turkish commented on <a href="https://medium.com/@enisteper/titanik-ke%C5%9Fifsel-veri-analizi-ve-makine-%C3%B6%C4%9Frenmesi-6ecc09d8df75">Medium</a>
* As English commented on <a href="https://www.kaggle.com/enisteper1/titanic-eda-machine-learning-top-3">Kaggle</a> 
* Full Code on <a href="https://github.com/enisteper1/Titanic-EDA-Machine-Learning">Github</a>

Note: I explained steps below as Turkish with images on <a href="https://medium.com/@enisteper/titanik-ke%C5%9Fifsel-veri-analizi-ve-makine-%C3%B6%C4%9Frenmesi-6ecc09d8df75">Medium</a>. Even I explained as Turkish it can be followed by keywords and all steps refer to each comment section at Medium.
## 1. Building a Server on AWS
**1.1** First, we log in to our account on AWS, if you do not have an account, it already takes a very short time to create one.

**1.2** Afterwards, we expand the All Services section in the AWS Management Console and click on the EC2 option under the Compute heading.

**1.3** Then, by clicking on the Launch instances section, we move on to creating a new server for ourselves. After going down a bit, we click on the Ubuntu 20.04 LTS 64-bit (x86) option. In order not to pay any fees, you must make sure that there is a Free tier eligible text under the server.

**1.4** After choosing Ubuntu, the page where we will choose the features of our server comes. Since we will continue for free, we continue with t2.micro, which is selected by default here.

**1.5** Before clicking Review and Launch button, we move on to the Configure Security Group heading to open our server to those who have ip.

**1.6** Under this heading, we choose **All traffic** for Type and **Anywhere** for Source. You can change the Security group name and description just like me so you can remember it later.

**1.7** After pressing **Review and Launch** and then **Launch**, we come to the last step, the key pair download part. In this part, we will download the key that we will connect to our server. Here you can give your key any name you want, I will prefer the name titanic. We create our server by clicking Download Key Pair button and then Launch instances. After the server is put into use, it will start working automatically. For this reason, you need to stop the server (instance) you have opened in case you take a break.

**1.8** You can click View instances to check your created server, or click EC2 in AWS Services and then click Instances under Instances.

## 2. Connecting to Server with PuTTy
**2.1** We will use <a href="https://www.putty.org/">PuTTy</a> to install the necessary libraries and our project on our server. If it is not installed on your computer, you can download the version suitable for your operating system from the link.

**2.2** After installing, open the application called PuTTygen from the windows search bar. After opening it, we need to find the key that we downloaded by clicking Load. Even if you go to the file where you downloaded the key, you may not see your key because it has a .pem extension. For this reason, you need to choose your key by changing the search option from *.ppk extension to *.*.

**2.3** After importing the key, you can save your *.ppk file with the name you specified by clicking **Save private key** and then yes. I save my own key as titanic.ppk. Now, if we switch to the part of connecting to our server, we need to first open the PuTTy application.
![image](https://user-images.githubusercontent.com/45767042/157896448-f3e6bca2-7e69-4aa8-948e-314ce00332bc.png)

**2.4** Here, we go back to the AWS site and click on the Instance ID link of our server that we have set up in the previous steps in the Instances section to obtain the IP address we need first.

**2.5** After entering the link, we copy the Public IPv4 address to connect with PuTTy. After copying, we go back to PuTTy and paste this address into the Host Name field. Afterwards, we click Data under Connection title on PuTTy and write the ubuntu name assigned by default when we create our server in the blank next to Auto-login username.

**2.6** After pressing the Open button to connect, we say Accept to the security alarm with the terminal that appears. We finally managed to connect to our server. :)

![image](https://user-images.githubusercontent.com/45767042/157897263-1c2488c6-cec2-4593-9abd-b3a5997fece2.png)

## 3. Downloading and Running the ML Based Web Project and 
**3.1** First, we write the update command to make the necessary installations of our server.

`sudo apt-get update && sudo apt-get upgrade`

**3.2** After completing the update and upgrade, we download our project with git.

`git clone https://github.com/enisteper1/AWS-Deployed-ML.git`

**3.3** After that, we move into the downloaded file and first pip3 and then we install the necessary libraries with the following commands.

```
cd AWS-Deployed-ML/
sudo apt install python3-pip
pip3 install -r requirements.txt
```

**3.4** Finally, we write the following command to run our project and you should see the output as I have shared below.

`python3 manage.py runserver 0.0.0.0:8000`

![image](https://user-images.githubusercontent.com/45767042/157897912-eb49883c-309e-4af9-ba01-9f49a1cc69ae.png)

**3.5** Now that we have our server running, we can move on to testing. For this, we first need to connect to our server over the web. For the connection link, we copy the IPv4 DNS by clicking on our server from the AWS instances section.

**3.6** After that, we open a new tab on the browser and paste the copied address and add :8000 to the end. I leave my own link below as an example.

`http://ec2-3-144-229-77.us-east-2.compute.amazonaws.com:8000/`

**3.7** For the trial step, I entered data by default. If you want, you can have the machine learning model predict whether this person survived the Titanic accident or not, with an experiment like I left below.
![image](https://user-images.githubusercontent.com/45767042/157898629-1be196a3-96c0-4de7-966e-8a0db4ae3785.png)

**3.8** Nice, we have deployed a machine learning model that runs on the server and can make a single prediction without the need to analyze all the data each time. Do not forget to stop your server from the options on the right by clicking the box to the left of your server from the AWS instances section after you have completed your trials.
![image](https://user-images.githubusercontent.com/45767042/157898355-7f7b80f9-c079-43a5-bc37-1122ad7a00f6.png)

Note: You can try browsing the Titanic dataset to observe how the data changes. You can find how the code works and how the parameters are created and saved from my project that in prediction folder.



