import React, { useEffect } from "react";
import jsPDF from "jspdf";

const PredictionReport = (props) => {
	useEffect(() => {
		// var doc = new jsPDF("landscape", "px", "a4", "false");
		// doc.text(60, 60, `Store Number : ${props.storeNum}`);
		// doc.text(60, 80, `Predicted Sales : ${props.predictedResult}`);
		// doc.save(`Store_${props.storeNum}.pdf`);

		var doc = new jsPDF("landscape", "px", "a4", "false");
		doc.setFillColor("#678283");
		doc.rect(0, 0, 800, 80, "F");
		//doc.setFontType("bold");
		doc.setFont("bold");
		doc.setFillColor("#FFFFFF");
		doc.roundedRect(60, 160, 100, 100, 5, 5, "F");
		doc.setTextColor("#ffffff");
		doc.setFontSize(25);
		doc.setFillColor("#161d24");
		doc.rect(0, 80, 800, 800, "F");
		doc.text(240, 120, "ASA Sales Forecasting");
		doc.setTextColor(0, 0, 0);
		doc.setFontSize(20);
		doc.setTextColor("#ffffff");

		doc.text(265, 140, "Prediction Report");
		doc.setTextColor(0, 0, 0);
		doc.setTextColor("#ffffff");

		doc.text(60, 180, `Store Number : ${props.storeNum}`);
		doc.text(60, 200, `Predicted Sales : ${props.predictedResult}`);
		doc.save(`Store_${props.storeNum}.pdf`);
	}, []);
	return <></>;
};

export default PredictionReport;
