import React, { useState, useEffect } from "react";
import { FiChevronRight, FiChevronLeft } from "react-icons/fi";
import { topSlider } from "./data";
// import { Link } from "react-router-dom";
// import Storeowner from "./Storeowner";
// import Company from "./Company";
import StoreLoginRegister from "./StoreLoginRegister";
import CompanyLoginRegister from "./CompanyLoginRegister";
import Report from "./Report";
const Home = () => {
	const [active, setActive] = useState("");
	const [slide, setSlide] = useState(topSlider);
	const [index, setIndex] = React.useState(0);

	useEffect(() => {
		const lastIndex = slide.length - 1;
		if (index < 0) {
			setIndex(lastIndex);
		}
		if (index > lastIndex) {
			setIndex(0);
		}
	}, [index, slide]);

	useEffect(() => {
		let slider = setInterval(() => {
			setIndex(index + 1);
		}, 5000);
		return () => {
			clearInterval(slider);
		};
	}, [index]);

	return (
		<div>
			<div id="home" style={{ paddingTop: "2rem" }}>
				<section className="section" style={{ width: "100%" }}>
					<div className="section-center" style={{ width: "100%" }}>
						{slide.map((sl, slIndex) => {
							const { id, image } = sl;

							let position = "nextSlide";
							if (slIndex === index) {
								position = "activeSlide";
							}
							if (
								slIndex === index - 1 ||
								(index === 0 && slIndex === slide.length - 1)
							) {
								position = "lastSlide";
							}

							return (
								<article
									className={position}
									key={id}
									style={{ width: "100%" }}
								>
									<img
										src={image}
										className="person-img"
										style={{ width: "100%" }}
									/>
								</article>
							);
						})}
						<button className="prev" onClick={() => setIndex(index - 1)}>
							<FiChevronLeft />
						</button>
						<button className="next" onClick={() => setIndex(index + 1)}>
							<FiChevronRight />
						</button>
					</div>
				</section>
				<div className="row" style={{ justifyContent: "center" }}>
					<div
						className="col-md-3 col-sm-3 col-xs-6"
						onClick={() => setActive("company")}
					>
						<button
							className="btn btn-sm animated-button thar-one"
							style={{
								borderStyle: "solid",
								borderColor: "#aca1ad",
								borderWidth: "3px",
							}}
						>
							<h5>COMPANY</h5>
						</button>
					</div>
					<div className="col-md-3 col-sm-3 col-xs-6">
						<button
							className="btn btn-sm animated-button thar-two"
							// onClick={(event) => (window.location.href = "/storeowner")}
							onClick={() => setActive("store")}
							style={{
								borderStyle: "solid",
								borderColor: "#aca1ad",
								borderWidth: "3px",
							}}
						>
							<h5>STORE</h5>
						</button>
					</div>
					<div className="col-md-3 col-sm-3 col-xs-6">
						<button
							// onClick={(event) => (window.location.href = "/report")}
							onClick={() => setActive("report")}
							className="btn btn-sm animated-button thar-three"
							style={{
								borderStyle: "solid",
								borderColor: "#aca1ad",
								borderWidth: "3px",
							}}
						>
							<h5>REPORTS</h5>
						</button>
					</div>
				</div>
			</div>
			{active === "store" && /*<Storeowner />*/ <StoreLoginRegister />}
			{active === "company" && /*<Company />*/ <CompanyLoginRegister />}
			{active === "report" && <Report />}
		</div>
	);
};

export default Home;
