import React, { useState, useRef } from "react";

import { links, social } from "./data";
import { Link } from "react-scroll";
import * as ReactBootStrap from "react-bootstrap";

const NavigationBar = () => {
	const [showLinks, setShowLinks] = useState(false);
	const linksContainerRef = useRef(null);
	const linksRef = useRef(null);
	const toggleLinks = () => {
		setShowLinks(!showLinks);
	};

	return (
		<ReactBootStrap.Navbar expand="lg" fixed="top">
			<ReactBootStrap.Navbar.Brand href="#home">
				<h4 style={{ paddingLeft: "1rem", color: "#311432" }}>
					ASA Sales Forecasting
				</h4>
			</ReactBootStrap.Navbar.Brand>
			<ReactBootStrap.Navbar.Toggle aria-controls="navbarScroll" />
			<ReactBootStrap.Navbar.Collapse
				id="navbarScroll"
				className="flex-row-reverse"
			>
				<ReactBootStrap.Nav>
					<section>
						<ul
							className="links"
							ref={linksRef}
							style={{ justifyContent: "center" }}
						>
							{links.map((link) => {
								const { id, url, text } = link;
								return (
									<Link
										to={text}
										spy={true}
										smooth={true}
										offset={-90}
										duration={500}
										key={id}
									>
										<ReactBootStrap.Nav.Link key={id}>
											<h6
												href={url}
												style={{ color: "#311432", paddingTop: "1rem" }}
											>
												{text}
											</h6>
										</ReactBootStrap.Nav.Link>
									</Link>
								);
							})}
						</ul>
					</section>
				</ReactBootStrap.Nav>
			</ReactBootStrap.Navbar.Collapse>
		</ReactBootStrap.Navbar>
	);
};

export default NavigationBar;
