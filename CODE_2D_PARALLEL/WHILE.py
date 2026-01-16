n = int(input("n = ?")) # on transforme en entier le chiffre rerntre par l'utilisateur ex: 346
compteur_de_colonne = 1 #on commence notre compte aux unites
while n > 0 : #tant quil  rest edes colonnes a compter 
    reste = int(n % 10) # ici on fait la division euclidienne de l'entier restant en elevant les colonnes deja comptees et deja imprimees ex: 6,
    print ("le ", compteur_de_colonne, "e chiffre de l'entier naturel saisi est le : ", reste)
    n = (n - reste )/10 #ici on va enlever le chiffre de la colonne comptee et l divise par 10 pour avoir celle de la colonne d'apres ex: 346 - 6 = 340 / 10 = 34
    compteur_de_colonne = compteur_de_colonne + 1 #je copte la colonne d'apres ex: apres 6 on comptera la 2 e colonne pour le 4
    
 
#2: function   
def somme_chiffre_entier(Entier):
    """
    Somme chiffre entier retourne la somme des chiffres composant l'entier 
    
    Args:
        - Entier saisi as an int
    Returns:
        - Somme as an int
    """
    n = int(Entier) #ex: 346
    somme = 0 #for now the sum is 0 
    while n > 0 : #WHILE there is a number to count
        somme = somme + int(n % 10) #la somme est egale a la somme enregistree + au reste de la division euclidienne ex : 0 + 6 + 4 + 3 = 13
        n = (n - reste )/10 #ici on va enlever le chiffre de la colonne comptee et l divise par 10 pour avoir celle de la colonne d'apres ex: 346 - 6 = 340 / 10 = 34
    
    
    return somme
  # 2: pour faire tourner le 2 : script qui appelle une fonction  
n = input("n = ? ")
somme = somme_chiffre_entier(n)
print("La somme des chiffres presents dans cet entier naturel est : ", somme)



# 3: somme ultime
def somme_ultime(Entier):
    """
    Somme ultime retourne la somme des chiffres composant l'entier jusqu'a ce qu'il n'y ai plus qu'un seul chiffre
    ex: 346: 3+4+6 = 13 : 1+3 = 4 
    
    Args:
        - Entier saisi as an int
    Returns:
        - Somme Ultime as an int
    """
    Entier = int(Entier) #ex: 346
    somme = somme_chiffre_entier(Entier) #for now the sum is 0 
    while somme > 9: #tant que la somme est superieure a 9, on veut  la reduire et refaisant la somme de ce qu'on trouve
        somme = somme_chiffre_entier(somme) #on refait la somme
    return somme
#script pour appeler la fonction
n = input("n = ? ")
somme_ultime = somme_ultime(n)
print("La somme ultime des chiffres presents dans cet entier naturel est : ", somme_ultime)