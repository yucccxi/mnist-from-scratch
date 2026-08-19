build:
	gcc -Iinclude -O3 -Wall -Wextra \
			\
			./src/compute_node.c     \
			./src/main.c ./src/mlp.c \
			./src/imath.c 			 \
			./src/mnist.c 			 \
			\
		-lm -o ./out/main

run:
	./out/main

clean:
	rm -f ./out/*