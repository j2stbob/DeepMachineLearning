import Deep_machine_learning_v2 as dml

def main():
    print(dml.model.training(50, 0.01))
    print((dml.model.predict([150, 0, 0, 1, 1, 27.5, 0.1])))
                    #Fare, Pclass_1, Pclass_2, Pclass_3, Sex, Age, Family size
                            #Select one Pclass you need and put 1 in the rest put 0



if __name__ == "__main__":
    main()