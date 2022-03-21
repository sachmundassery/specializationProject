import React, { useState } from "react";

const ContactUs = () => {
	const [contactUs, setContactUs] = useState({
		name: "",
		email: "",
		message: "",
	});

	const handleChange = (e) => {
		const input = e.target.name;
		const value = e.target.value;
		console.log(input, value);
		setContactUs({ ...contactUs, [input]: value });
	};

	const handleSubmit = async (e) => {
		e.preventDefault();
		const formData = new FormData();

		formData.append("email", contactUs.email);
		formData.append("name", contactUs.name);
		formData.append("message", contactUs.message);
		const response = await fetch("http://127.0.0.1:8000/contactus", {
			method: "POST",
			body: formData,
		});
		const result = await response.json();

		if (result["response"] == true) {
			console.log("response successfully reached");
			alert("Message Received, Thankyou !!");
		}
	};

	return (
		<div id="contact">
			<main>
				<section
					className="container"
					style={{ paddingLeft: "0px", paddingRight: "0px" }}
				>
					<div class="contact_head">
						<h3>Contact Us</h3>
						{/* </div> */}
						{/* <label
						htmlFor="name"
						style={{ paddingLeft: "0px", paddingRight: "0px" }}
					>
						Name
					</label> */}
						{/* <div class="contact_head"> */}
						<input
							type="text"
							name="name"
							id="name"
							value={contactUs.name}
							onChange={handleChange}
							placeholder="Name"
						/>

						<br />

						<input
							type="email"
							name="email"
							id="email"
							value={contactUs.email}
							onChange={handleChange}
							placeholder="Email"
						/>

						<br />

						<input
							type="text"
							name="message"
							id="message"
							value={contactUs.message}
							onChange={handleChange}
							placeholder="Message"
						/>

						<br />

						<button
							className="printAndSend"
							type="button"
							onClick={handleSubmit}
						>
							Post
						</button>

						<br />
					</div>
				</section>
			</main>
		</div>
	);
};

export default ContactUs;
