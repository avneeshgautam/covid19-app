from flask import Flask,render_template,request
import pickle
app = Flask(__name__)

file = open("model.pkl","rb")
classifier = pickle.load(file)
file.close()


@app.route('/' ,methods=["GET","POST"])
def hello_world():
    if request.method == "POST":
        print(request.form)

        myDict = request.form
        fever = float(int(myDict['fever']))
        age = int(myDict['age'])
        bodypain = int(myDict['bodypain'])
        cold = int(myDict['cold'])
        breath = int(myDict['breath'])
        
        features=[fever,bodypain,age,cold,breath]
        # features =[100,1,2,3,0]
        inf_prob = classifier.predict_proba([features])[0][1]
        # print(inf_prob)
        return render_template('show.html',inf=round(inf_prob*100))
    return render_template('index.html')

    # return "inf_prob"+str(inf_prob)
    
if __name__ == "__main__":
    app.run(debug=True)