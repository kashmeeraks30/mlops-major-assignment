from utils import load_dataset,load_model,train_model,model_metrics,save_model

def main():
    X,y=load_dataset()
    model,X_test,y_test = train_model(X,y)
    model = load_model()
    model_metrics(model,X_test,y_test)

if __name__=="__main__":
    main()