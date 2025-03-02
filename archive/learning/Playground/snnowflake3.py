import pygame
import random

# Define constants
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 0)
PARTICLE_COLOR = (255, 255, 255)
PARTICLE_RADIUS = 1
GRAVITY = 2

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Define the particle class
class Particle:
    def __init__(self):
        self.x = random.randrange(WIDTH)
        self.y = 0
        self.speed = random.uniform(1, 3)

    def move(self, shape):
        self.y += self.speed * GRAVITY

        if shape.collidepoint(self.x, self.y + PARTICLE_RADIUS):
            self.y = shape.top - PARTICLE_RADIUS
            self.speed = 0

    def draw(self):
        pygame.draw.circle(screen, PARTICLE_COLOR, (int(self.x), int(self.y)), PARTICLE_RADIUS)

# Initialize particles
particles = [Particle() for _ in range(5000)]

# Initialize invisible shape
shape = pygame.Rect(200, 200, 200, 50)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLOR)

    # Change the position of the invisible shape randomly
    shape.x = random.randint(0, WIDTH - shape.width)
    shape.y = random.randint(0, HEIGHT - shape.height)

    for particle in particles:
        particle.move(shape)
        particle.draw()

    # Remove particles that have hit the bottom
    particles = [p for p in particles if p.y < HEIGHT]

    # Add new particles
    while len(particles) < 5000:
        particles.append(Particle())

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

