package com.example.accountapp;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

@ComponentScan("com.example")
@SpringBootApplication
public class Account1Application {

	public static void main(String[] args) {
		SpringApplication.run(Account1Application.class, args);
	}

}
