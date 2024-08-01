#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
import pickle
import os

class person_info:
    def __init__(self):
        self.person_db=[]
        self.secret_dates = []
        self.data_file="problem1_data_file.pickle"
     
     # method for getting the inputs from the user   
    def get_input(self):

        name=input("\nEnter the person name:")
        dob=input("Enter the dob in dd/mm/yyyy:")
        
        if self.is_valid_date_format(dob):
            secret_dob=input("Mark this as secret DOB:(yes/no):").strip().lower()
            if secret_dob=="yes":
                self.secret_dates.append(dob)
                print("DOB is marked as secret")
                self.person_db.append((name,dob))
            else:
                self.person_db.append((name,dob))
                self.save_file()
        else:
            print("Invalid date format. Please enter dob in dd/mm/yyyy format.")
    
    # method for validating the date format    
    def is_valid_date_format(self, dob):
        try:
            datetime.strptime(dob, '%d/%m/%Y')
            return True
        except ValueError:
            return False
        
    #method for displaying the personal info for a particular person    
    def display_dob(self):
        name=input("\nEnter the person's name:")
        for person in self.person_db:
            if person[0] == name:
                if person[1] in self.secret_dates:
                    print("Date of Birth is Secret for this person, Try another person")
                else:
                    print("Date of Birth:", person[1])
                return
        print("\nPerson name not found.")
        
    def save_file(self):
        try:
            with open("c:\\Users\\epvin\\OneDrive\\Desktop\\Week 4\\problem1_data_file.pickle", 'wb') as file:
                 pickle.dump(self.person_db, file)
            print("Data saved successfully.")
            #print("\nContents of self.person_db:\n", self.person_db) 
        except IOError as e:
            print(f"Error saving file: {e}")
        
        
    def load_file(self):
        try:
            if os.path.exists(self.data_file):  
                 with open("c:\\Users\\epvin\\OneDrive\\Desktop\\Week 4\\problem1_data_file.pickle", "rb") as file:
                    self.person_db = pickle.load(file)
                 print("Data loaded successfully.")
            else:
                print("No existing data file found")
        except IOError as e:
            print(f"Error while loading:{e}")
        
    
def main():
    person_file = person_info()
    person_file.load_file()
    while True:
        print("\n-----Managing Personal Info______")
        print("\n1. Adding a Person's name and DOB")
        print("2. Displaying the Date of Birth for a particular person")
        print("3. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            person_file.get_input()
        elif choice == "2":
            person_file.display_dob()
        elif choice == "3":
            print("....Exiting...")
            
            break
        else:
            print("Pick a choice from above. Please try again.")
                  
if __name__=="__main__":          
    main()

