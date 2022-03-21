import React from "react";

const About = () => {
	return (
		<div id="about" style={{ textAlign: "center" }}>
			<b>
				<h3
					style={{ textAlign: "center", paddingTop: "50px", color: "#DADADA" }}
				>
					About
				</h3>
			</b>
			<p
				style={{
					color: "#DADADA",
					fontSize: "20px",
					padding: "40px 100px 20px 100px",
					textAlign: "center",
					display: "block",
				}}
			>
				The project titled 'ASA Sales Forecasting,' is a solution to an
				organization's growth. Startups and MNCs' are increasing day by day.
				Therefore, accurate sales forecasting contributes to a vital impact on
				business. A layman might wish to see organizational growth as per the
				current trend. A store owner might want to see the predicted sales if he
				or she increases the promotion or number of working days. A company
				might wish to see its current sales trend. Then this project would help
				them out. Many of such existing sales prediction websites and software
				are not free, but this one is.
			</p>
		</div>
	);
};

export default About;
