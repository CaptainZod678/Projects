package com.example.accountapp.exception;

import org.junit.Test;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(value = HttpStatus.NOT_FOUND)
public class AccountNotFoundException extends RuntimeException{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private String accountName;
	private String profileId;
	private Object accountValue;
	
	public AccountNotFoundException(String accountName, String profileId, Object accountvalue) {
		super(String.format("Account id %s not found with %s: %s",accountName, profileId, accountvalue));
		this.accountName = accountName;
		this.profileId = profileId;
		this.accountValue = accountvalue;
	}
	
	public String getAccountName() {
		return accountName;
	}
	
	public String getProfileId() {
		return profileId;
	}

	public Object getAccountValue() {
		return accountValue;
	}

}
