import React, { useState } from "react";

const Company = (props) => {
	const handleUpload = async (e) => {
		const fileObj = e.target.files[0];
		const fileName = fileObj.name;
		document.querySelector(".file-name").textContent = fileName;
		const userLoggedin = props.userLoggedin;
		console.log("company.js", userLoggedin);

		alert("File Uploaded !");

		const formData = new FormData();
		console.log(fileObj);
		formData.append("type", "csvUpload");
		formData.append("userName", userLoggedin);
		formData.append("csvUploaded", fileObj);

		const response = await fetch("http://127.0.0.1:8000/company", {
			method: "POST",
			body: formData,
		});
		const result = await response.json();
	};

	return (
		<form className="form-wrapper">
			<h3>Upload your dataset</h3>

			<input
				type="file"
				accept=".csv,.xlsx,.xls"
				name="file"
				id="file"
				className="inputfile"
				onChange={handleUpload}
			/>
			<label htmlFor="file">
				<strong>Choose a file</strong>
			</label>
			<br />
			<p
				style={{ paddingTop: "1rem", color: "black" }}
				className="file-name"
			></p>
		</form>
	);
};

export default Company;
