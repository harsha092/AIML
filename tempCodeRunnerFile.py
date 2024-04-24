def add_random_cities(self):
        num_cities = simpledialog.askinteger("Random Cities", "Enter the number of random cities to add:", initialvalue=5)
        if num_cities is not None:
            for _ in range(num_cities):
                x, y = random.randint(20, 380), random.randint(20, 380)
                self.canvas.create_oval(x - self.city_size, y - self.city_size,
                                        x + self.city_size, y + self.city_size, fill='black')
                self.cities.append((x, y))
            self.update_buttons_state()  # Enable solve and clear buttons
            self.draw_tour([])  # Clear previous tour
            messagebox.showinfo("Random Cities Added", f"{num_cities} random cities added successfully.")