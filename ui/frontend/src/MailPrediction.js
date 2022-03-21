import React, { useEffect } from "react";
import emailjs from "emailjs-com";

const MailPrediction = (props) => {
	const sendEmail = (e) => {
		e.preventDefault();

		var templateParams = {
			storeNum: props.storeNum,
			predictedSales: props.predictedResult[0],
			email: props.emailRequired,
		};
		console.log("-------", templateParams.predictedSales);

		emailjs
			.send(
				"service_nb0git7",
				"template_bfd9z9o",
				templateParams,
				"user_nbjVbNVYTDvYbnpQxoasQ"
			)
			.then((res) => {
				console.log(res);
			})
			.catch((err) => console.log(err));
	};
	return (
		<div className="btnDiv">
			<button onClick={sendEmail} className="printAndSend">
				Send
			</button>
		</div>
	);
};

export default MailPrediction;
