import Deep_machine_lerning as dml


def main():
    dml.model.training(200, 0.001)        # you can change step(2arg) for speed
    print((dml.model.predict([310, 1, 0, 24]))) # you can add a "round" of func to find a accurate the answer xd(joke) You can add more values but change little bit code in dml
                            # Fare, Pclass, Sex, Age

if __name__ == "__main__":
    main()

# there may be a error, just download pandas 2.2.0
# its model is too capitalist xd


# ⠀      ⠀⠀⠀⠀⠀⠀⠀⡟⣭⣭⣷⣶⣶⣾⣿⣶⣭⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⠙⠛⠛⠛⠛⠿⠟⠻⠿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠬⠓⠒⠶⠶⠦⠴⠶⠶⣚⣥⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀  ⠀⠀⠀⠀⠀⠀⠀⠈⢉⣟⣀⡀⠉⠒⠃⣉⣈⡳⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⣾⣿⣿⣦⣄⠀⠀⠀⣀⣀⠤⠭⠶⠉⠉⠉⠉⠓⠒⠯⢤⣀⠀⠀⠀⠀⣴⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⢹⡛⠛⠋⠈⠓⢄⡎⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⣀⡀⠀⠈⠳⡞⠉⠚⠈⡩⠝⠛⠃⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠑⠤⠤⢤⡙⡾⠀⠀⢀⣴⠀⠠⠌⡁⠀⠀⠩⠄⣟⡦⢤⣀⣹⡤⠤⠤⠇⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣇⡀⠀⠈⠙⠧⣄⠀⠀⠀⠀⠀⠀⣿⣿⣿⣶⣤⣹⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠲⢤⣀⣀⣙⣆⠀⠀⡀⠄⣿⣿⣿⣿⣿⣿⣿⣧⣀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡞⠀⠀⠀⢱⡌⠀⠀⠘⢦⠉⠛⠻⢿⣿⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡼⠀⠀⠀⠀⡎⢱⠀⠀⠀⠘⡆⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⠀⢀⡴⠃⠘⢦⠀⠀⠀⢹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⣸⠀⠀⠀⠀⣇⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⣠⠎⠀⠀⠀⠀⠀⠈⠳⢄⣸⠀⠀