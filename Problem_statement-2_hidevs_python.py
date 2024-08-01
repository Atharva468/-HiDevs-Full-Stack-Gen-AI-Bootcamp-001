#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pickle
import re

class address_book:
    
    def __init__(self):
        self.addressbook_db=[]  
        
    def save_file(self):
            with open("C:\\Users\\epvin\\OneDrive\\Desktop\\Week 4\\problem2_data_file.pickle", 'wb') as file:
                 pickle.dump(self.addressbook_db, file)
            
        
        
    def load_file(self):
        try:
            with open("C:\\Users\\epvin\\OneDrive\\Desktop\\Week 4\\problem2_data_file.pickle", "rb") as file:
                self.addressbook_db = pickle.load(file)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("No existing data file found. Starting with empty address book.")

        
    def get_input(self):
         print("\n***********Addressbook database**********")
         self.Fname = input("Enter the first name:")
         self.Lname = input("\nEnter the last name:")
         self.StreetAddress = input("\nEnter the street address:")
         self.City = input("\nEnter the city:")
         self.State = input("\nEnter the state:")
         self.Country = input("\nEnter the country:")
         contact_details = (self.Fname,self.Lname,self.StreetAddress,self.City,self.State,self.Country)
         self.addressbook_db.append((self.Fname,self.Lname,self.StreetAddress,self.City,self.State,self.Country))
         self.save_file()

    def is_valid_email(self,email):
    # Regular expression pattern for validating email addresses
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None 

    def num_occurrences(self,Fname,Lname,StreetAddress):
        count = 0
        for num in self.addressbook_db:
        # Check if all three criteria match: first name, last name, and street address
             if num[0] == Fname and num[1] == Lname and num[2] == StreetAddress:
                count += 1
        return count
    
    
    def main(self): 
         unique_contacts= set()
         while True:

        #Adding Email Address
            email= input("\nEnter the Email address:")
            if not self.is_valid_email(email):
                 print("Invalid email format. Please enter a valid email address.")
                 continue       
            elif email in unique_contacts:
                print("Email already exits,Try another email")
                continue
            else:
                unique_contacts.add(email)
                
        #Adding Phone Number    
            phone_number=input("\nEnter the phone number: ")
            if phone_number in unique_contacts:
                print("Phone Number already exits,Try different one")
                continue
            unique_contacts.add(phone_number)
            print("\n****Contact added successfully****")
            print("First name:",self.Fname,
                  "\nLast name:",self.Lname,
                  "\nStreet Address:",self.StreetAddress,
                  "\nCity:",self.City,
                  "\nState:",self.State,
                  "\nCountry:",self.Country,
                 "\nEmail:",email,
                 "\nPhone number:",phone_number)
            self.save_file()

            add_another = input("Do you want to add another contact? (yes/no): ")
            if add_another.lower() != 'yes':
                print("-------Exiting-----")
                find_occur = input("Do you want to find the no of occurences of a contact:? (yes/no): ")
                if find_occur.lower()=='yes':
                    print("\n***********Finding No of occurrences of a contact in Addressbook database**********")
                    self.Fname = input("Enter the first name:")
                    self.Lname = input("\nEnter the last name:")
                    self.StreetAddress = input("\nEnter the street address:")
                    occurrences = addressbook_file.num_occurrences(self.Fname,self.Lname,self.StreetAddress)
                    print(f"\nOccurrences of firstname '{self.Fname}',lastname '{self.Lname}', streetaddress'{self.StreetAddress}': {occurrences}") 
                break
            else:
                self.get_input()

addressbook_file=address_book()
addressbook_file.load_file()  # Load existing data
addressbook_file.get_input()
addressbook_file.main()


# In[ ]:




