# définition des cibles particulières
.PHONY: clean, mrproper
 
# désactivation des règles implicites
.SUFFIXES:

CFFLAGS= -W -Wall -ansi -pedantic -g -w

SRC= $(wildcard *.cpp)
OBJ= $(SRC:.cpp=.o)

FastHough: $(OBJ)
	@g++ -o $@ $^ `pkg-config opencv --libs`

%.o : %.cpp
	@g++ -o $@ -c $< $(CFFLAGS) -Wall `pkg-config opencv --cflags`
# édition de liens
	
clean:
	rm -rf *.o
mrproper: clean
	rm -rf FastHough